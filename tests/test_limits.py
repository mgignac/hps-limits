"""Tests for hps_limits."""

import json
import tempfile
from pathlib import Path

import pytest
import pyhf

from hps_limits import (
    SearchInput,
    build_model,
    compute_upper_limit,
    compute_upper_limit_from_input,
    scan_mass_limits,
    scan_mass_limits_from_input,
)
from hps_limits.limits import LimitResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_SEARCH = {
    "regions": {
        "region_a": {
            "background": 50.0,
            "observed": 48,
            "systematics": [
                {
                    "name": "bkg_stat_a",
                    "type": "shapesys",
                    "samples": ["background"],
                    "value": 5.0,
                }
            ],
        },
        "region_b": {
            "background": 30.0,
            "observed": 29,
            "systematics": [
                {
                    "name": "bkg_stat_b",
                    "type": "shapesys",
                    "samples": ["background"],
                    "value": 3.0,
                }
            ],
        },
    }
}

SIGNALS_A = {"region_a": 5.0, "region_b": 3.0}


# ---------------------------------------------------------------------------
# SearchInput
# ---------------------------------------------------------------------------


class TestSearchInput:
    def test_from_dict(self):
        s = SearchInput.from_dict(SIMPLE_SEARCH)
        assert s.region_names == ["region_a", "region_b"]

    def test_from_json(self, tmp_path):
        p = tmp_path / "search.json"
        p.write_text(json.dumps(SIMPLE_SEARCH))
        s = SearchInput.from_json(p)
        assert s.region_names == ["region_a", "region_b"]

    def test_missing_regions_key_raises(self):
        with pytest.raises(ValueError, match="'regions'"):
            SearchInput.from_dict({})

    def test_empty_regions_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SearchInput.from_dict({"regions": {}})

    def test_missing_background_raises(self):
        with pytest.raises(ValueError, match="'background'"):
            SearchInput.from_dict(
                {"regions": {"r": {"observed": 10, "systematics": []}}}
            )

    def test_missing_observed_raises(self):
        with pytest.raises(ValueError, match="'observed'"):
            SearchInput.from_dict(
                {"regions": {"r": {"background": 10.0, "systematics": []}}}
            )

    def test_zero_background_raises(self):
        with pytest.raises(ValueError, match="positive"):
            SearchInput.from_dict(
                {"regions": {"r": {"background": 0.0, "observed": 5}}}
            )

    def test_unsupported_syst_type_raises(self):
        with pytest.raises(ValueError, match="unsupported systematic type"):
            SearchInput.from_dict(
                {
                    "regions": {
                        "r": {
                            "background": 10.0,
                            "observed": 10,
                            "systematics": [
                                {
                                    "name": "x",
                                    "type": "badtype",
                                    "samples": ["background"],
                                    "value": 1.0,
                                }
                            ],
                        }
                    }
                }
            )

    def test_unknown_sample_raises(self):
        with pytest.raises(ValueError, match="unknown sample"):
            SearchInput.from_dict(
                {
                    "regions": {
                        "r": {
                            "background": 10.0,
                            "observed": 10,
                            "systematics": [
                                {
                                    "name": "x",
                                    "type": "normsys",
                                    "samples": ["data"],
                                    "value": 0.05,
                                }
                            ],
                        }
                    }
                }
            )

    def test_to_pyhf_model_returns_model_and_data(self):
        s = SearchInput.from_dict(SIMPLE_SEARCH)
        model, data = s.to_pyhf_model(SIGNALS_A)
        assert isinstance(model, pyhf.Model)
        assert len(data) > 0

    def test_to_pyhf_model_missing_signal_raises(self):
        s = SearchInput.from_dict(SIMPLE_SEARCH)
        with pytest.raises(ValueError, match="missing for region"):
            s.to_pyhf_model({"region_a": 5.0})  # region_b missing

    def test_normsys_correlated_across_regions(self):
        """Same syst name in two regions → one shared NP in the model."""
        data = {
            "regions": {
                "r1": {
                    "background": 50.0,
                    "observed": 50,
                    "systematics": [
                        {
                            "name": "lumi",
                            "type": "normsys",
                            "samples": ["signal"],
                            "value": 0.03,
                        }
                    ],
                },
                "r2": {
                    "background": 30.0,
                    "observed": 30,
                    "systematics": [
                        {
                            "name": "lumi",
                            "type": "normsys",
                            "samples": ["signal"],
                            "value": 0.03,
                        }
                    ],
                },
            }
        }
        s = SearchInput.from_dict(data)
        model, _ = s.to_pyhf_model({"r1": 5.0, "r2": 3.0})
        # "lumi" should appear exactly once in the NP list
        assert model.config.par_names.count("lumi") == 1

    def test_shapesys_uncorrelated_across_regions(self):
        """Different syst names → independent NPs (pyhf appends '[0]' for shapesys)."""
        s = SearchInput.from_dict(SIMPLE_SEARCH)
        model, _ = s.to_pyhf_model(SIGNALS_A)
        par_names = model.config.par_names
        assert any(p.startswith("bkg_stat_a") for p in par_names)
        assert any(p.startswith("bkg_stat_b") for p in par_names)


# ---------------------------------------------------------------------------
# build_model (simple API)
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_single_channel(self):
        model, data = build_model(
            signal=5.0, background=50.0, background_uncertainty=5.0
        )
        assert isinstance(model, pyhf.Model)

    def test_multi_channel(self):
        model, _ = build_model(
            signal=[5.0, 3.0],
            background=[50.0, 30.0],
            background_uncertainty=[5.0, 3.0],
        )
        assert isinstance(model, pyhf.Model)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            build_model(
                signal=[5.0, 3.0],
                background=[50.0],
                background_uncertainty=[5.0],
            )

    def test_default_observed_equals_background(self):
        _, data = build_model(signal=5.0, background=50.0, background_uncertainty=5.0)
        assert data[0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# compute_upper_limit (simple API)
# ---------------------------------------------------------------------------


class TestComputeUpperLimit:
    def test_returns_limit_result(self):
        result = compute_upper_limit(
            signal=5.0, background=50.0, background_uncertainty=5.0
        )
        assert isinstance(result, LimitResult)

    def test_mass_stored(self):
        result = compute_upper_limit(
            signal=5.0,
            background=50.0,
            background_uncertainty=5.0,
            observed=50.0,
            mass=30.0,
        )
        assert result.mass == 30.0

    def test_expected_band_has_five_values(self):
        result = compute_upper_limit(
            signal=10.0, background=100.0, background_uncertainty=10.0
        )
        assert len(result.mu_upper_exp_band) == 5

    def test_band_ordering(self):
        result = compute_upper_limit(
            signal=10.0, background=100.0, background_uncertainty=10.0
        )
        b = result.mu_upper_exp_band
        assert b[0] < b[1] < b[2] < b[3] < b[4]

    def test_large_signal_gives_tight_limit(self):
        result = compute_upper_limit(
            signal=1000.0,
            background=10.0,
            background_uncertainty=1.0,
            observed=10.0,
        )
        assert result.mu_upper_obs < 0.1

    def test_cls_obs_in_unit_interval(self):
        result = compute_upper_limit(
            signal=5.0, background=50.0, background_uncertainty=5.0
        )
        assert 0.0 <= result.cls_obs <= 1.0


# ---------------------------------------------------------------------------
# compute_upper_limit_from_input
# ---------------------------------------------------------------------------


class TestComputeUpperLimitFromInput:
    def test_returns_limit_result(self):
        s = SearchInput.from_dict(SIMPLE_SEARCH)
        result = compute_upper_limit_from_input(s, SIGNALS_A, mass=30.0)
        assert isinstance(result, LimitResult)
        assert result.mass == 30.0

    def test_limit_values_positive(self):
        s = SearchInput.from_dict(SIMPLE_SEARCH)
        result = compute_upper_limit_from_input(s, SIGNALS_A)
        assert result.mu_upper_obs > 0
        assert result.mu_upper_exp > 0

    def test_band_ordering(self):
        s = SearchInput.from_dict(SIMPLE_SEARCH)
        result = compute_upper_limit_from_input(s, SIGNALS_A)
        b = result.mu_upper_exp_band
        assert b[0] < b[1] < b[2] < b[3] < b[4]

    def test_with_normsys_on_signal(self):
        data = {
            "regions": {
                "r": {
                    "background": 50.0,
                    "observed": 50,
                    "systematics": [
                        {
                            "name": "lumi",
                            "type": "normsys",
                            "samples": ["signal"],
                            "value": 0.05,
                        }
                    ],
                }
            }
        }
        s = SearchInput.from_dict(data)
        result = compute_upper_limit_from_input(s, {"r": 5.0})
        assert isinstance(result, LimitResult)


# ---------------------------------------------------------------------------
# scan_mass_limits / scan_mass_limits_from_input
# ---------------------------------------------------------------------------


class TestScanMassLimits:
    def test_returns_one_result_per_mass(self):
        masses = [20.0, 25.0, 30.0]
        results = scan_mass_limits(
            masses=masses,
            signals=[5.0, 4.0, 3.0],
            backgrounds=[50.0, 40.0, 30.0],
            background_uncertainties=[5.0, 4.0, 3.0],
        )
        assert len(results) == 3
        for result, mass in zip(results, masses):
            assert result.mass == mass

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            scan_mass_limits(
                masses=[20.0, 25.0],
                signals=[5.0],
                backgrounds=[50.0, 40.0],
                background_uncertainties=[5.0, 4.0],
            )


class TestScanMassLimitsFromInput:
    def test_returns_one_result_per_mass(self):
        s = SearchInput.from_dict(SIMPLE_SEARCH)
        signals_by_mass = {
            20.0: {"region_a": 8.0, "region_b": 5.0},
            25.0: {"region_a": 7.0, "region_b": 4.0},
            30.0: {"region_a": 6.0, "region_b": 3.0},
        }
        results = scan_mass_limits_from_input(s, signals_by_mass)
        assert len(results) == 3

    def test_mass_values_preserved(self):
        s = SearchInput.from_dict(SIMPLE_SEARCH)
        signals_by_mass = {
            20.0: SIGNALS_A,
            30.0: SIGNALS_A,
        }
        results = scan_mass_limits_from_input(s, signals_by_mass)
        assert results[0].mass == 20.0
        assert results[1].mass == 30.0
