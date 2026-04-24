"""Input format for HPS search results.

The JSON format is::

    {
        "regions": {
            "<region_name>": {
                "background": <float>,
                "observed": <float>,
                "systematics": [
                    {
                        "name": "<name>",
                        "type": "normsys" | "shapesys",
                        "samples": ["signal"] | ["background"] | ["signal", "background"],
                        "value": <float>
                    }
                ]
            }
        }
    }

Systematic types
----------------
normsys
    Log-normal constraint on sample normalisation.  ``value`` is the relative
    (fractional) 1σ uncertainty, e.g. ``0.03`` for 3 %.
shapesys
    Poisson-constrained per-bin uncertainty.  ``value`` is the *absolute* 1σ
    uncertainty on the expected yield.  Typically used for background
    statistical uncertainties.

Correlation convention
----------------------
Systematics with the **same name** are treated as correlated across regions
(they share one nuisance parameter in the global fit).  Use distinct names
for uncorrelated uncertainties (e.g. ``"bkg_stat_region_a"`` vs
``"bkg_stat_region_b"``).

Signal yields are **not** part of the JSON — they depend on the A′ mass
hypothesis being tested and are supplied separately when computing limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pyhf

if TYPE_CHECKING:
    pass

_SUPPORTED_SYST_TYPES = frozenset({"normsys", "shapesys"})
_SUPPORTED_SAMPLES = frozenset({"signal", "background"})


class SearchInput:
    """HPS search result: background estimates, observed data, and systematics.

    Construct via :meth:`from_json` or :meth:`from_dict`.  Signal yields are
    provided separately at limit-setting time so that the same search
    configuration can be reused across many A′ mass hypotheses.
    """

    def __init__(self, data: dict) -> None:
        self._data = data
        self._validate()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "SearchInput":
        """Load from a JSON file."""
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def from_dict(cls, data: dict) -> "SearchInput":
        """Create from a plain Python dictionary."""
        return cls(data)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def region_names(self) -> list[str]:
        """Ordered list of region names."""
        return list(self._data["regions"].keys())

    def to_pyhf_spec(self, signals: dict[str, float]) -> dict:
        """Build a pyhf workspace specification.

        Parameters
        ----------
        signals:
            Expected signal yield in each region, keyed by region name.
            Every region defined in this input must have a corresponding entry.

        Returns
        -------
        dict
            A pyhf-compatible workspace dict (internal format — users should
            not need to interact with this directly).
        """
        missing = set(self.region_names) - set(signals)
        if missing:
            raise ValueError(
                f"Signal yields missing for region(s): {sorted(missing)}"
            )

        channels = []
        observations = []

        for rname in self.region_names:
            region = self._data["regions"][rname]
            sig = float(signals[rname])
            bkg = float(region["background"])
            obs = float(region["observed"])

            sig_modifiers: list[dict] = [
                {"name": "mu", "type": "normfactor", "data": None}
            ]
            bkg_modifiers: list[dict] = []

            for syst in region.get("systematics", []):
                mod = _build_modifier(syst)
                if "signal" in syst["samples"]:
                    sig_modifiers.append(mod)
                if "background" in syst["samples"]:
                    bkg_modifiers.append(mod)

            channels.append(
                {
                    "name": rname,
                    "samples": [
                        {
                            "name": "signal",
                            "data": [sig],
                            "modifiers": sig_modifiers,
                        },
                        {
                            "name": "background",
                            "data": [bkg],
                            "modifiers": bkg_modifiers,
                        },
                    ],
                }
            )
            observations.append({"name": rname, "data": [obs]})

        return {
            "channels": channels,
            "observations": observations,
            "measurements": [
                {"name": "HPS", "config": {"poi": "mu", "parameters": []}}
            ],
            "version": "1.0.0",
        }

    def to_pyhf_model(
        self, signals: dict[str, float]
    ) -> tuple[pyhf.Model, list[float]]:
        """Build a pyhf Model and data vector.

        Parameters
        ----------
        signals:
            Expected signal yield per region.

        Returns
        -------
        (model, data)
        """
        spec = self.to_pyhf_spec(signals)
        ws = pyhf.Workspace(spec)
        model = ws.model()
        data = ws.data(model)
        return model, data

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if "regions" not in self._data:
            raise ValueError("Input must contain a top-level 'regions' key.")
        if not self._data["regions"]:
            raise ValueError("'regions' must not be empty.")

        for rname, region in self._data["regions"].items():
            for key in ("background", "observed"):
                if key not in region:
                    raise ValueError(
                        f"Region '{rname}' is missing required field '{key}'."
                    )
            if float(region["background"]) <= 0:
                raise ValueError(
                    f"Region '{rname}': 'background' must be positive."
                )

            for syst in region.get("systematics", []):
                for key in ("name", "type", "samples", "value"):
                    if key not in syst:
                        raise ValueError(
                            f"Systematic in region '{rname}' missing field '{key}'."
                        )
                if syst["type"] not in _SUPPORTED_SYST_TYPES:
                    raise ValueError(
                        f"Region '{rname}': unsupported systematic type "
                        f"'{syst['type']}'. Supported: {sorted(_SUPPORTED_SYST_TYPES)}."
                    )
                bad_samples = set(syst["samples"]) - _SUPPORTED_SAMPLES
                if bad_samples:
                    raise ValueError(
                        f"Region '{rname}', systematic '{syst['name']}': "
                        f"unknown sample(s) {bad_samples}. "
                        f"Supported: {sorted(_SUPPORTED_SAMPLES)}."
                    )
                if not syst["samples"]:
                    raise ValueError(
                        f"Region '{rname}', systematic '{syst['name']}': "
                        "'samples' must not be empty."
                    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _build_modifier(syst: dict) -> dict:
    """Convert a systematic entry to a pyhf modifier dict."""
    name = syst["name"]
    value = float(syst["value"])

    if syst["type"] == "normsys":
        # Log-normal constraint; value is the relative 1σ uncertainty
        return {
            "name": name,
            "type": "normsys",
            "data": {"hi": 1.0 + value, "lo": 1.0 - value},
        }
    elif syst["type"] == "shapesys":
        # Poisson constraint; value is the absolute 1σ uncertainty (single bin)
        return {
            "name": name,
            "type": "shapesys",
            "data": [value],
        }
    else:
        raise ValueError(f"Unsupported systematic type: {syst['type']}")


# ---------------------------------------------------------------------------
# ABCD background JSON loader
# ---------------------------------------------------------------------------


@dataclass
class BackgroundEntry:
    """Background estimate at a single (region, mass) point from an ABCD result.

    Attributes
    ----------
    region:
        Detector region name (e.g. ``"L1L1"``, ``"L2L2"``).
    mass_mev:
        A′ mass hypothesis in MeV.
    nbkg:
        Background estimate in the signal window.
    nbkg_err:
        1σ uncertainty on ``nbkg``.
    observed:
        Observed count in the signal window (region A of the ABCD plane).
    observed_err:
        Statistical uncertainty on ``observed``.
    resolution_mev:
        Invariant-mass resolution (MeV) at this mass.
    signal_window_mev:
        Full width of the signal window (MeV).
    """

    region: str
    mass_mev: float
    nbkg: float
    nbkg_err: float
    observed: float
    observed_err: float
    resolution_mev: float
    signal_window_mev: float


@dataclass
class BackgroundEstimate:
    """Collection of ABCD background estimates loaded from a results JSON.

    Attributes
    ----------
    meta:
        Metadata from the JSON (luminosity, z-window definitions, etc.).
    entries:
        Flat list of :class:`BackgroundEntry` objects, one per
        (region, mass) combination, in the order they appear in the file.
    """

    meta: dict
    entries: list[BackgroundEntry] = field(default_factory=list)

    def by_mass(self) -> dict[float, list[BackgroundEntry]]:
        """Return entries grouped by mass (MeV), sorted ascending."""
        from collections import defaultdict
        groups: dict[float, list[BackgroundEntry]] = defaultdict(list)
        for e in self.entries:
            groups[e.mass_mev].append(e)
        return dict(sorted(groups.items()))

    def by_region(self) -> dict[str, list[BackgroundEntry]]:
        """Return entries grouped by region name."""
        from collections import defaultdict
        groups: dict[str, list[BackgroundEntry]] = defaultdict(list)
        for e in self.entries:
            groups[e.region].append(e)
        return dict(groups)

    @property
    def masses_mev(self) -> list[float]:
        """Sorted unique mass values (MeV) present in the estimate."""
        return sorted({e.mass_mev for e in self.entries})

    @property
    def regions(self) -> list[str]:
        """Unique region names present in the estimate (order of first appearance)."""
        seen: dict[str, None] = {}
        for e in self.entries:
            seen[e.region] = None
        return list(seen)


def load_background_json(
    path: str | Path,
    projection: str | None = None,
) -> BackgroundEstimate:
    """Load ABCD background estimates from a results JSON file.

    Parameters
    ----------
    path:
        Path to the ``results.json`` file produced by the ABCD background
        estimator.
    projection:
        If ``None`` (default), use the background estimates as-is
        (``bkg_est`` / ``bkg_est_err``).  Pass the name of a projection key
        (e.g. ``"100% projection"``) to substitute that projection's estimates
        instead.  Raises ``KeyError`` if the projection name is not present for
        a given entry.

    Returns
    -------
    BackgroundEstimate
    """
    with open(path) as f:
        raw = json.load(f)

    meta = raw.get("meta", {})
    entries: list[BackgroundEntry] = []

    for region_name, mass_points in raw.get("regions", {}).items():
        for pt in mass_points:
            if projection is not None:
                proj = pt["projections"][projection]
                nbkg = float(proj["bkg_est"])
                nbkg_err = float(proj["bkg_est_err"])
            else:
                nbkg = float(pt["bkg_est"])
                nbkg_err = float(pt["bkg_est_err"])

            entries.append(
                BackgroundEntry(
                    region=region_name,
                    mass_mev=round(float(pt["mass_GeV"]) * 1000, 6),
                    nbkg=nbkg,
                    nbkg_err=nbkg_err,
                    observed=float(pt["n_a_obs"]),
                    observed_err=float(pt["n_a_obs_err"]),
                    resolution_mev=float(pt["resolution_GeV"]) * 1000,
                    signal_window_mev=float(pt["signal_window_GeV"]) * 1000,
                )
            )

    return BackgroundEstimate(meta=meta, entries=entries)


# ---------------------------------------------------------------------------
# Signal JSON loader
# ---------------------------------------------------------------------------


@dataclass
class SignalEntry:
    """Expected signal yield at a single (region, mass, ε²) point.

    Attributes
    ----------
    region:
        Detector region name (e.g. ``"L1L1"``, ``"L2L2"``).
    mass_mev:
        A′ mass hypothesis in MeV.
    eps2:
        ε² (kinetic-mixing coupling squared) for this signal point.
    nsig:
        Expected signal yield in the signal window.
    acceptance:
        Signal acceptance at this point (optional, for bookkeeping).
    """

    region: str
    mass_mev: float
    eps2: float
    nsig: float
    acceptance: float | None = None


@dataclass
class SignalEstimate:
    """Collection of signal yields loaded from a signal JSON file.

    Attributes
    ----------
    meta:
        Metadata from the JSON (luminosity, process description, etc.).
    entries:
        Flat list of :class:`SignalEntry` objects, one per
        (region, mass, ε²) combination, in the order they appear in the file.
    """

    meta: dict
    entries: list[SignalEntry] = field(default_factory=list)

    def by_mass(self) -> dict[float, list[SignalEntry]]:
        """Return entries grouped by mass (MeV), sorted ascending."""
        from collections import defaultdict
        groups: dict[float, list[SignalEntry]] = defaultdict(list)
        for e in self.entries:
            groups[e.mass_mev].append(e)
        return dict(sorted(groups.items()))

    def by_region(self) -> dict[str, list[SignalEntry]]:
        """Return entries grouped by region name."""
        from collections import defaultdict
        groups: dict[str, list[SignalEntry]] = defaultdict(list)
        for e in self.entries:
            groups[e.region].append(e)
        return dict(groups)

    @property
    def masses_mev(self) -> list[float]:
        """Sorted unique mass values (MeV) present in the estimate."""
        return sorted({e.mass_mev for e in self.entries})

    @property
    def regions(self) -> list[str]:
        """Unique region names (order of first appearance)."""
        seen: dict[str, None] = {}
        for e in self.entries:
            seen[e.region] = None
        return list(seen)

    @property
    def eps2_values(self) -> list[float]:
        """Sorted unique ε² values present in the estimate."""
        return sorted({e.eps2 for e in self.entries})


def load_signal_json(path: str | Path) -> SignalEstimate:
    """Load signal yield estimates from a JSON file.

    The JSON format mirrors the background results JSON but with an added
    ``epsilon_squared`` field per entry and ``nsig`` in place of background
    fields::

        {
            "meta": { ... },
            "regions": {
                "<region_name>": [
                    {
                        "mass_GeV": <float>,
                        "epsilon_squared": <float>,
                        "nsig": <float>,
                        "acceptance": <float>   (optional)
                    },
                    ...
                ]
            }
        }

    Parameters
    ----------
    path:
        Path to the signal JSON file.

    Returns
    -------
    SignalEstimate
    """
    with open(path) as f:
        raw = json.load(f)

    meta = raw.get("meta", {})
    entries: list[SignalEntry] = []

    for region_name, points in raw.get("regions", {}).items():
        for pt in points:
            entries.append(
                SignalEntry(
                    region=region_name,
                    mass_mev=round(float(pt["mass_GeV"]) * 1000, 6),
                    eps2=float(pt["epsilon_squared"]),
                    nsig=float(pt["nsig"]),
                    acceptance=float(pt["acceptance"]) if "acceptance" in pt else None,
                )
            )

    return SignalEstimate(meta=meta, entries=entries)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def load_search_csv(path: str | Path) -> list[dict]:
    """Load a signal/background CSV file as a flat list of (mass, ε²) points.

    The CSV must have columns::

        mass_gev, epsilon_squared, Nsig, Nbkg

    Each row is one point in the 2D (mass, ε²) search space.  The returned
    list has one entry per row, preserving the full grid structure.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    list of dict
        Each dict has:

        ``"mass_mev"`` — A′ mass in MeV (converted from GeV).
        ``"eps2"``     — ε² value for this point.
        ``"nsig"``     — Expected signal yield at this (mass, ε²).
        ``"nbkg"``     — Expected background yield for this mass window.
    """
    import numpy as np

    data = np.genfromtxt(path, delimiter=",", names=True)

    points = []
    for row in data:
        points.append({
            "mass_mev": round(float(row["mass_gev"]) * 1000, 6),
            "eps2":     float(row["epsilon_squared"]),
            "nsig":     float(row["Nsig"]),
            "nbkg":     float(row["Nbkg"]),
        })

    return points
