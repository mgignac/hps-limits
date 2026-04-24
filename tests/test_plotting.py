"""Tests for hps_limits.plotting."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, must be set before pyplot import

import numpy as np
import pytest

from hps_limits import SearchInput, scan_mass_limits_from_input
from hps_limits.plotting import compute_cls_scan, plot_cls_scan, plot_exclusion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEARCH = {
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

SIGNALS = {"region_a": 5.0, "region_b": 3.0}
EPS2_REF = 1e-6

SIGNALS_BY_MASS = {
    20.0: {"region_a": 8.0, "region_b": 5.0},
    25.0: {"region_a": 7.0, "region_b": 4.0},
    30.0: {"region_a": 6.0, "region_b": 3.0},
}
EPS2_BY_MASS = {m: EPS2_REF for m in SIGNALS_BY_MASS}


@pytest.fixture(scope="module")
def scan_results():
    s = SearchInput.from_dict(SEARCH)
    return scan_mass_limits_from_input(s, SIGNALS_BY_MASS)


@pytest.fixture(scope="module")
def cls_scan_output():
    s = SearchInput.from_dict(SEARCH)
    return compute_cls_scan(s, SIGNALS, poi_range=(0.0, 5.0), n_points=10)


# ---------------------------------------------------------------------------
# compute_cls_scan
# ---------------------------------------------------------------------------


class TestComputeClsScan:
    def test_returns_three_items(self, cls_scan_output):
        result = cls_scan_output
        assert len(result) == 3

    def test_mu_values_shape(self, cls_scan_output):
        mu_values, _, _ = cls_scan_output
        assert len(mu_values) == 10

    def test_cls_obs_shape(self, cls_scan_output):
        _, cls_obs, _ = cls_scan_output
        assert len(cls_obs) == 10

    def test_cls_exp_band_has_five_arrays(self, cls_scan_output):
        _, _, cls_exp_band = cls_scan_output
        assert len(cls_exp_band) == 5
        for arr in cls_exp_band:
            assert len(arr) == 10

    def test_cls_obs_in_unit_interval(self, cls_scan_output):
        _, cls_obs, _ = cls_scan_output
        assert np.all(cls_obs >= 0.0)
        assert np.all(cls_obs <= 1.0)

    def test_cls_obs_decreasing(self, cls_scan_output):
        """CLs should generally decrease as mu increases."""
        _, cls_obs, _ = cls_scan_output
        # Not strictly monotone due to numerics, but first > last
        assert cls_obs[0] >= cls_obs[-1]

    def test_exp_band_ordering_at_interior_points(self, cls_scan_output):
        """At fixed μ > 0: −2σ band has lower CLs than +2σ (−2σ is more sensitive).

        μ = 0 is excluded because all bands converge to ~1 with arbitrary ordering.
        """
        mu_values, _, band = cls_scan_output
        for i, mu in enumerate(mu_values):
            if mu == 0.0:
                continue
            vals = [band[j][i] for j in range(5)]
            assert vals[0] <= vals[2] <= vals[4], (
                f"band ordering failed at μ={mu:.2f} (index {i}): {vals}"
            )


# ---------------------------------------------------------------------------
# plot_exclusion
# ---------------------------------------------------------------------------


class TestPlotExclusion:
    def test_returns_axes(self, scan_results):
        ax = plot_exclusion(scan_results)
        import matplotlib.axes
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_with_eps2_ref(self, scan_results):
        ax = plot_exclusion(scan_results, eps2_ref=EPS2_REF)
        assert r"\varepsilon" in ax.get_ylabel()

    def test_with_eps2_by_mass(self, scan_results):
        ax = plot_exclusion(scan_results, eps2_by_mass=EPS2_BY_MASS)
        assert r"\varepsilon" in ax.get_ylabel()

    def test_without_eps2_shows_mu(self, scan_results):
        ax = plot_exclusion(scan_results)
        assert r"\mu" in ax.get_ylabel()

    def test_no_band(self, scan_results):
        ax = plot_exclusion(scan_results, show_band=False)
        # Should still produce a valid axes with two lines (obs + exp)
        import matplotlib.axes
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_yscale_log(self, scan_results):
        ax = plot_exclusion(scan_results, eps2_ref=EPS2_REF)
        assert ax.get_yscale() == "log"

    def test_custom_style(self, scan_results):
        ax = plot_exclusion(
            scan_results,
            eps2_ref=EPS2_REF,
            obs_style={"color": "blue"},
            exp_style={"color": "red"},
        )
        import matplotlib.axes
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# plot_cls_scan
# ---------------------------------------------------------------------------


class TestPlotClsScan:
    def test_returns_axes(self):
        s = SearchInput.from_dict(SEARCH)
        ax = plot_cls_scan(s, SIGNALS, poi_range=(0.0, 5.0), n_points=10)
        import matplotlib.axes
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_ylabel_is_cls(self):
        s = SearchInput.from_dict(SEARCH)
        ax = plot_cls_scan(s, SIGNALS, poi_range=(0.0, 5.0), n_points=10)
        assert "CL" in ax.get_ylabel()

    def test_xlabel_with_eps2_ref(self):
        s = SearchInput.from_dict(SEARCH)
        ax = plot_cls_scan(s, SIGNALS, eps2_ref=EPS2_REF,
                           poi_range=(0.0, 5.0), n_points=10)
        assert r"\varepsilon" in ax.get_xlabel()

    def test_xlabel_without_eps2_ref(self):
        s = SearchInput.from_dict(SEARCH)
        ax = plot_cls_scan(s, SIGNALS, poi_range=(0.0, 5.0), n_points=10)
        assert r"\mu" in ax.get_xlabel()

    def test_title_includes_mass(self):
        s = SearchInput.from_dict(SEARCH)
        ax = plot_cls_scan(s, SIGNALS, mass=30.0,
                           poi_range=(0.0, 5.0), n_points=10)
        assert "30.0" in ax.get_title()

    def test_ylim_zero_to_one(self):
        s = SearchInput.from_dict(SEARCH)
        ax = plot_cls_scan(s, SIGNALS, poi_range=(0.0, 5.0), n_points=10)
        assert ax.get_ylim() == (0.0, 1.0)
