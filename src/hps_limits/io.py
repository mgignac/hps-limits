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

            res_gev = pt.get("resolution_GeV")
            # When a projection is used the actual n_a_obs no longer matches
            # the scaled background, so use the Asimov (obs = nbkg) instead.
            if projection is not None:
                observed = nbkg
                observed_err = nbkg_err
            else:
                observed = float(pt["n_a_obs"])
                observed_err = float(pt["n_a_obs_err"])

            entries.append(
                BackgroundEntry(
                    region=region_name,
                    mass_mev=round(float(pt["mass_GeV"]) * 1000, 6),
                    nbkg=nbkg,
                    nbkg_err=nbkg_err,
                    observed=observed,
                    observed_err=observed_err,
                    resolution_mev=float(res_gev) * 1000 if res_gev is not None else float("nan"),
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
# Combined results JSON loader
# ---------------------------------------------------------------------------


def load_results_json(
    path: str | Path,
    regions: list[str] | None = None,
    projection: str | None = None,
) -> tuple[BackgroundEstimate, SignalEstimate]:
    """Load background and signal from a single combined ABCD results JSON.

    The file format has a top-level ``meta`` dict and a ``regions`` dict whose
    values are lists of mass-point entries.  Each entry carries background
    fields (``bkg_est``, ``n_a_obs``, …) **and** a ``signal`` sub-dict keyed
    by ``"ap_{mass_MeV}_eps2_{eps2}"`` strings::

        {
            "meta": { ... },
            "regions": {
                "<region>": [
                    {
                        "mass_GeV": <float>,
                        "bkg_est": <float>,
                        "bkg_est_err": <float>,
                        "n_a_obs": <float>,
                        "n_a_obs_err": <float>,
                        "signal": {
                            "ap_60_eps2_1.000e-07": {
                                "epsilon_sq": <float>,
                                "nsig": <float>,
                                "acceptance": <float>,
                                ...
                            },
                            ...
                        }
                    },
                    ...
                ]
            }
        }

    For each mass-point entry only the signal hypotheses whose A′ mass matches
    the window mass are extracted (on-peak signal).

    Parameters
    ----------
    path:
        Path to the combined results JSON file.
    regions:
        Region names to load.  Defaults to all regions *except* ``"inclusive"``,
        which is the incoherent aggregate.  Pass an explicit list to override.
    projection:
        Optional projection key inside ``pt["projections"]`` to substitute
        for the default ``bkg_est`` / ``bkg_est_err`` fields.

    Returns
    -------
    (BackgroundEstimate, SignalEstimate)
    """
    with open(path) as f:
        raw = json.load(f)

    meta = raw.get("meta", {})
    available = list(raw.get("regions", {}).keys())

    if regions is None:
        regions = [r for r in available if r != "inclusive"]

    bg_entries: list[BackgroundEntry] = []
    sig_entries: list[SignalEntry] = []

    for region_name in regions:
        mass_points = raw["regions"].get(region_name, [])
        for pt in mass_points:
            mass_mev = round(float(pt["mass_GeV"]) * 1000, 6)

            # --- background ---
            if projection is not None:
                proj = pt["projections"][projection]
                nbkg = float(proj["bkg_est"])
                nbkg_err = float(proj["bkg_est_err"])
            else:
                nbkg = float(pt["bkg_est"])
                nbkg_err = float(pt["bkg_est_err"])

            res_gev = pt.get("resolution_GeV")
            if projection is not None:
                observed = nbkg
                observed_err = nbkg_err
            else:
                observed = float(pt["n_a_obs"])
                observed_err = float(pt["n_a_obs_err"])

            bg_entries.append(BackgroundEntry(
                region=region_name,
                mass_mev=mass_mev,
                nbkg=nbkg,
                nbkg_err=nbkg_err,
                observed=observed,
                observed_err=observed_err,
                resolution_mev=float(res_gev) * 1000 if res_gev is not None else float("nan"),
                signal_window_mev=float(pt["signal_window_GeV"]) * 1000,
            ))

            # --- signal (on-peak only: A′ mass == window mass) ---
            ap_prefix = f"ap_{round(mass_mev)}_"
            for sig_key, sig_val in pt.get("signal", {}).items():
                if not sig_key.startswith(ap_prefix):
                    continue
                sig_entries.append(SignalEntry(
                    region=region_name,
                    mass_mev=mass_mev,
                    eps2=float(sig_val["epsilon_sq"]),
                    nsig=float(sig_val["nsig"]),
                    acceptance=float(sig_val["acceptance"]),
                ))

    bg = BackgroundEstimate(meta=meta, entries=bg_entries)
    sig = SignalEstimate(meta=meta, entries=sig_entries)
    return bg, sig


# ---------------------------------------------------------------------------
# Cut-scan JSON loader  (flat list, single region, epsilon not epsilon_sq)
# ---------------------------------------------------------------------------


def _region_from_suffix(suffix: str) -> str:
    """Extract region name from a hit-category tag.

    Handles both ``'isL1L1_cut'`` and ``'isL1L1'`` (with or without a trailing
    ``_cut``) → ``'L1L1'``.
    """
    import re
    m = re.search(r"is([A-Za-z0-9]+?)(?:_cut)?$", suffix)
    return m.group(1) if m else "signal_region"


def load_cut_scan_json(
    path: str | Path,
    region: str | None = None,
) -> tuple[BackgroundEstimate, SignalEstimate]:
    """Load background and signal from a cut-scan results JSON.

    The file format is a flat list of grid points::

        {
            "metadata": { "suffix": "isL1L1_cut", ... },
            "n_points": <int>,
            "data": [
                {
                    "mass_MeV": <float>,
                    "epsilon":  <float>,      # ε (not ε²) — squared internally
                    "signal":   <float>,      # expected signal yield
                    "background": <float>,    # background estimate
                    "Zbi":      <float>
                },
                ...
            ]
        }

    Background is the same for all ε at a given mass; a single
    :class:`BackgroundEntry` is created per mass.  Since the file carries no
    observed count, ``background`` is used as the Asimov observed.
    The background uncertainty defaults to ``sqrt(background)`` (Poisson stat).

    Parameters
    ----------
    path:
        Path to the cut-scan JSON file.
    region:
        Region name to assign.  If ``None``, extracted from
        ``metadata["suffix"]`` (e.g. ``"isL1L1_cut"`` → ``"L1L1"``).

    Returns
    -------
    (BackgroundEstimate, SignalEstimate)
    """
    import math

    with open(path) as f:
        raw = json.load(f)

    meta = raw.get("metadata", {})

    if region is None:
        region = _region_from_suffix(meta.get("suffix", ""))

    data = raw.get("data", [])

    # Group by mass so we emit one BackgroundEntry per mass
    from collections import defaultdict
    by_mass: dict[float, list[dict]] = defaultdict(list)
    for pt in data:
        by_mass[float(pt["mass_MeV"])].append(pt)

    bg_entries: list[BackgroundEntry] = []
    sig_entries: list[SignalEntry] = []

    for mass_mev in sorted(by_mass):
        pts = by_mass[mass_mev]
        # background is the same for all epsilon at this mass
        nbkg = float(pts[0]["background"])
        nbkg_err = math.sqrt(nbkg) if nbkg > 0 else 1e-10

        bg_entries.append(BackgroundEntry(
            region=region,
            mass_mev=mass_mev,
            nbkg=nbkg,
            nbkg_err=nbkg_err,
            observed=nbkg,        # Asimov: no real observed in this format
            observed_err=math.sqrt(max(nbkg, 1.0)),
            resolution_mev=float("nan"),
            signal_window_mev=float("nan"),
        ))

        for pt in pts:
            eps = float(pt["epsilon"])
            sig_entries.append(SignalEntry(
                region=region,
                mass_mev=mass_mev,
                eps2=eps * eps,
                nsig=float(pt["signal"]),
                acceptance=None,
            ))

    bg = BackgroundEstimate(meta=meta, entries=bg_entries)
    sig = SignalEstimate(meta=meta, entries=sig_entries)
    return bg, sig


def load_cut_scan_jsons(
    paths: list[str | Path],
) -> tuple[BackgroundEstimate, SignalEstimate]:
    """Load background and signal from multiple cut-scan JSON files.

    Each file contributes one detector region (the region name is extracted
    from ``metadata["suffix"]`` in each file).  The entries are merged into
    a single :class:`BackgroundEstimate` and :class:`SignalEstimate` suitable
    for multi-channel limit setting.

    Different files may have different mass and ε grids; the scan functions
    handle the intersection automatically.

    Parameters
    ----------
    paths:
        Ordered list of cut-scan JSON file paths (one per hit category).

    Returns
    -------
    (BackgroundEstimate, SignalEstimate)
    """
    all_bg: list[BackgroundEntry] = []
    all_sig: list[SignalEntry] = []
    combined_meta: dict = {}

    for path in paths:
        bg, sig = load_cut_scan_json(path)
        all_bg.extend(bg.entries)
        all_sig.extend(sig.entries)
        if not combined_meta:
            combined_meta = dict(bg.meta)

    return (
        BackgroundEstimate(meta=combined_meta, entries=all_bg),
        SignalEstimate(meta=combined_meta, entries=all_sig),
    )


# ---------------------------------------------------------------------------
# z0-binned cut-scan JSON loader  (multi-bin shape fit in min_y0)
# ---------------------------------------------------------------------------


def load_z0_binned_cut_scan_json(
    path: str | Path,
    region: str | None = None,
) -> tuple[BackgroundEstimate, SignalEstimate]:
    """Load background and signal from a z0-binned cut-scan results JSON.

    Extends the flat cut-scan format with per-z0-bin yields.  Each z0 bin
    becomes a separate pyhf channel named ``{hit_category}_{bin_label}``
    (e.g. ``"L1L1_A"``, ``"L1L1_B"``, ``"L1L1_C"``), enabling a shape fit
    in the fit variable (``min_y0``)::

        {
            "metadata": {
                "suffix": "isL1L1_cut",
                "z0_regions": {
                    "A": {"lo": 0.24, "hi": 0.38},
                    "B": {"lo": 0.38, "hi": 0.48},
                    "C": {"lo": 0.48, "hi": null}
                },
                ...
            },
            "data": [
                {
                    "mass_MeV":       <float>,
                    "epsilon":        <float>,
                    "signal_regionA": <float>,
                    "background_regionA": <float>,
                    "signal_regionB": <float>,
                    "background_regionB": <float>,
                    ...
                },
                ...
            ]
        }

    Background is constant across ε at a given mass.  Since the file carries
    no observed count, the background estimate is used as the Asimov observed.
    Background uncertainty defaults to ``sqrt(background)`` (Poisson stat).

    Parameters
    ----------
    path:
        Path to the z0-binned cut-scan JSON file.
    region:
        Hit-category prefix for channel names.  If ``None``, extracted from
        ``metadata["suffix"]`` (e.g. ``"isL1L1_cut"`` → ``"L1L1"``).

    Returns
    -------
    (BackgroundEstimate, SignalEstimate)
    """
    import math
    from collections import defaultdict

    with open(path) as f:
        raw = json.load(f)

    meta = raw.get("metadata", {})

    if region is None:
        region = _region_from_suffix(meta.get("suffix") or meta.get("hit_category", ""))

    z0_regions = meta.get("z0_regions", {})
    if not z0_regions:
        raise ValueError(
            f"No 'z0_regions' key found in metadata of {path}. "
            "Use load_cut_scan_json for single-bin files."
        )
    bin_keys = list(z0_regions.keys())

    data = raw.get("data", [])

    by_mass: dict[float, list[dict]] = defaultdict(list)
    for pt in data:
        by_mass[float(pt["mass_MeV"])].append(pt)

    bg_entries: list[BackgroundEntry] = []
    sig_entries: list[SignalEntry] = []

    # Detect which fields are present — files may be signal-only or background-only.
    first_pt_any = next(iter(by_mass.values()))[0] if by_mass else {}
    has_background = any(
        f"background_region{k}" in first_pt_any for k in bin_keys
    )
    has_signal = any(
        f"signal_region{k}" in first_pt_any for k in bin_keys
    )

    for mass_mev in sorted(by_mass):
        pts = by_mass[mass_mev]
        first_pt = pts[0]  # background is the same for all ε at this mass

        for bin_key in bin_keys:
            channel = f"{region}_{bin_key}"

            if has_background:
                nbkg = float(first_pt[f"background_region{bin_key}"])
                nbkg_err = math.sqrt(nbkg) if nbkg > 0 else 1e-10
                bg_entries.append(BackgroundEntry(
                    region=channel,
                    mass_mev=mass_mev,
                    nbkg=max(nbkg, 1e-10),
                    nbkg_err=nbkg_err,
                    observed=nbkg,
                    observed_err=math.sqrt(max(nbkg, 1.0)),
                    resolution_mev=float("nan"),
                    signal_window_mev=float("nan"),
                ))

            if has_signal:
                for pt in pts:
                    eps = float(pt["epsilon"])
                    nsig = float(pt[f"signal_region{bin_key}"])
                    sig_entries.append(SignalEntry(
                        region=channel,
                        mass_mev=mass_mev,
                        eps2=eps * eps,
                        nsig=nsig,
                        acceptance=None,
                    ))

    return (
        BackgroundEstimate(meta=meta, entries=bg_entries),
        SignalEstimate(meta=meta, entries=sig_entries),
    )


def load_z0_binned_cut_scan_jsons(
    paths: list[str | Path],
) -> tuple[BackgroundEstimate, SignalEstimate]:
    """Load and merge z0-binned cut-scan JSONs from multiple hit-category files.

    Each file contributes one hit category with its z0 sub-bins as separate
    channels.  Entries from all files are merged into a single
    :class:`BackgroundEstimate` and :class:`SignalEstimate` suitable for
    multi-channel limit setting.

    Parameters
    ----------
    paths:
        Ordered list of z0-binned cut-scan JSON file paths (one per hit
        category, e.g. L1L1, L1L2, L2L2, L2L3).

    Returns
    -------
    (BackgroundEstimate, SignalEstimate)
    """
    all_bg: list[BackgroundEntry] = []
    all_sig: list[SignalEntry] = []
    combined_meta: dict = {}

    for path in paths:
        bg, sig = load_z0_binned_cut_scan_json(path)
        all_bg.extend(bg.entries)
        all_sig.extend(sig.entries)
        if not combined_meta:
            combined_meta = dict(bg.meta)

    return (
        BackgroundEstimate(meta=combined_meta, entries=all_bg),
        SignalEstimate(meta=combined_meta, entries=all_sig),
    )


# ---------------------------------------------------------------------------
# A′ displaced-vertex signal JSON loader
# ---------------------------------------------------------------------------


def load_ap_signal_jsons(
    entries: list[tuple[str | Path, str]],
) -> SignalEstimate:
    """Load and merge multiple ap_displaced signal JSONs.

    Parameters
    ----------
    entries:
        List of ``(path, region)`` pairs, one per hit category.

    Returns
    -------
    SignalEstimate
        All entries merged into a single estimate.
    """
    all_entries: list[SignalEntry] = []
    for path, region in entries:
        sig = load_ap_signal_json(path, region=region)
        all_entries.extend(sig.entries)
    return SignalEstimate(meta={}, entries=all_entries)


def synthetic_zero_background(sig: SignalEstimate) -> BackgroundEstimate:
    """Create a zero-background estimate covering the signal's (region, mass) grid.

    Used when ``zero_background: true`` and no background file is provided.
    Every background value and observed count is 0; ``apply_zero_background``
    in the driver is a no-op on top of this.
    """
    entries: list[BackgroundEntry] = []
    seen: set[tuple[str, float]] = set()
    for e in sig.entries:
        key = (e.region, e.mass_mev)
        if key not in seen:
            seen.add(key)
            entries.append(BackgroundEntry(
                region=e.region,
                mass_mev=e.mass_mev,
                nbkg=0.0,
                nbkg_err=0.0,
                observed=0.0,
                observed_err=0.0,
                resolution_mev=float("nan"),
                signal_window_mev=float("nan"),
            ))
    return BackgroundEstimate(meta={}, entries=entries)


def load_ap_signal_json(
    path: str | Path,
    region: str = "L1L1",
) -> SignalEstimate:
    """Load A′ displaced-vertex signal yields from a mass-keyed JSON.

    The file format is a flat dict keyed by ``"ap{mass_MeV}MeV"`` strings.
    Each value has parallel ``eps2`` and ``yields`` arrays of equal length::

        {
            "ap70MeV":  {"eps2": [1e-10, ...], "yields": [2.6e-6, ...]},
            "ap80MeV":  {"eps2": [...],         "yields": [...]},
            ...
        }

    Parameters
    ----------
    path:
        Path to the signal JSON file.
    region:
        Region name to assign to all entries (default ``"L1L1"``).

    Returns
    -------
    SignalEstimate
    """
    import re as _re

    with open(path) as f:
        raw = json.load(f)

    entries: list[SignalEntry] = []

    for key in sorted(raw.keys()):
        m = _re.match(r"ap(\d+(?:\.\d+)?)MeV", key)
        if m is None:
            continue
        mass_mev = float(m.group(1))
        data = raw[key]
        for eps2, nsig in zip(data["eps2"], data["yields"]):
            entries.append(SignalEntry(
                region=region,
                mass_mev=mass_mev,
                eps2=float(eps2),
                nsig=float(nsig),
                acceptance=None,
            ))

    return SignalEstimate(meta={}, entries=entries)


# ---------------------------------------------------------------------------
# z0-binned background loader from results JSON (z_binned field)
# ---------------------------------------------------------------------------


def load_results_z0_background_json(
    path: str | Path,
    bin_label: str,
    projection: str | None = None,
) -> BackgroundEstimate:
    """Load a single z0-bin background file in standard ABCD results format.

    Each z0 bin is stored as a separate file (e.g. ``results_L1L1_bin1.json``,
    ``results_L1L1_bin2.json``).  The file format is identical to the standard
    ABCD results JSON (``bkg_est``, ``n_a_obs``, ``projections``).  The region
    names in the returned estimate are suffixed with ``_<bin_label>`` so they
    match the signal channel names (e.g. ``L1L1_lo``, ``L1L1_hi``).

    Parameters
    ----------
    path:
        Path to the single-bin results JSON file.
    bin_label:
        Label suffix to append to every region name (e.g. ``"lo"`` or ``"hi"``).
    projection:
        Optional projection key (e.g. ``"10% dataset"``).  When given, the
        projected ``bkg_est`` and ``bkg_est_err`` values are used instead of
        the raw 1%-dataset values.  The observed count is set to the projected
        background (Asimov).

    Returns
    -------
    BackgroundEstimate
        Region names follow the pattern ``{region}_{bin_label}``.
    """
    import re as _re
    bg = load_background_json(path, projection=projection)
    for e in bg.entries:
        # Strip a trailing proj-significance selection tag (e.g. "L1L1_ps17"
        # → "L1L1") so the channel matches the signal's hit-category name.
        base = _re.sub(r"_ps\d+$", "", e.region)
        e.region = f"{base}_{bin_label}"
    return bg


def load_results_z0_background_jsons(
    paths: list[str | Path],
    projection: str | None = None,
    bin_labels: list[str] | None = None,
) -> BackgroundEstimate:
    """Load and merge z0-bin background files in standard ABCD results format.

    Each file represents one z0 bin for one or more hit categories.  The bin
    label is derived from the ``_bin{N}`` suffix in the filename, mapped to
    ``bin_labels[N-1]`` (default: ``["lo", "hi"]``).

    Parameters
    ----------
    paths:
        List of per-bin results JSON file paths (e.g. all ``*_bin*.json`` files).
    projection:
        Passed to :func:`load_results_z0_background_json`.
    bin_labels:
        Mapping from 1-based bin number to label string.
        Defaults to ``["lo", "hi"]`` (bin1 → "lo", bin2 → "hi").

    Returns
    -------
    BackgroundEstimate
    """
    import re as _re

    _labels = bin_labels if bin_labels is not None else ["lo", "hi"]

    all_entries: list[BackgroundEntry] = []
    combined_meta: dict = {}

    for path in paths:
        stem = Path(path).stem
        m = _re.search(r"_bin(\d+)", stem)
        if m is None:
            raise ValueError(
                f"Cannot determine z0 bin number from filename '{stem}'. "
                "Expected '_bin1', '_bin2', etc. in the filename."
            )
        bin_idx = int(m.group(1)) - 1  # 1-based → 0-based
        if bin_idx >= len(_labels):
            raise ValueError(
                f"Bin index {bin_idx + 1} exceeds bin_labels list "
                f"(length {len(_labels)}): {_labels}"
            )
        label = _labels[bin_idx]
        bg = load_results_z0_background_json(path, bin_label=label,
                                             projection=projection)
        all_entries.extend(bg.entries)
        if not combined_meta:
            combined_meta = dict(bg.meta)

    return BackgroundEstimate(meta=combined_meta, entries=all_entries)


# ---------------------------------------------------------------------------
# ABCD background-systematics loader
# ---------------------------------------------------------------------------


def load_background_systematics_json(
    path: str | Path,
    bin_labels: list[str] | None = None,
) -> dict[str, dict]:
    """Load ABCD background-estimate systematics keyed by pyhf channel name.

    The file (e.g. ``background_systematics.json``) stores, under a top-level
    ``systematics`` dict, one entry per ``{region}_bin{N}`` key (e.g.
    ``"L1L1_bin1"``).  Each entry carries fractional 1σ shifts of the ABCD
    background prediction::

        "L1L1_bin1": {
            "region": "L1L1",
            "components_frac": {"sideband": 0.005452, "control_region": 0.000711},
            "total_frac": 0.005498,
            ...
        }

    The ``_bin{N}`` index is mapped to the same z0 bin label used by the
    background loader (default ``bin1 → "lo"``, ``bin2 → "hi"``) so the returned
    keys match the model channel names (``L1L1_lo``, ``L1L1_hi``).

    Parameters
    ----------
    path:
        Path to the background-systematics JSON file.
    bin_labels:
        1-based bin-number → label mapping.  Defaults to ``["lo", "hi"]`` to
        match :func:`load_results_z0_background_jsons`.

    Returns
    -------
    dict
        ``{channel_name: {"region": str, "components_frac": {name: frac},
        "total_frac": float}}``.  Only the fields needed to build pyhf normsys
        modifiers are retained.
    """
    import re as _re

    _labels = bin_labels if bin_labels is not None else ["lo", "hi"]

    with open(path) as f:
        raw = json.load(f)

    systs = raw.get("systematics", {})
    out: dict[str, dict] = {}

    for key, entry in systs.items():
        m = _re.search(r"_bin(\d+)$", key)
        if m is None:
            # No z0-bin suffix → single-channel (non-binned) region: the key
            # (or its "region" field) is the channel name directly, e.g. "L1L1".
            region = entry.get("region") or key
            channel = region
        else:
            bin_idx = int(m.group(1)) - 1
            if bin_idx >= len(_labels):
                raise ValueError(
                    f"Bin index {bin_idx + 1} in '{key}' exceeds bin_labels "
                    f"(length {len(_labels)}): {_labels}"
                )
            region = entry.get("region") or key[: m.start()]
            channel = f"{region}_{_labels[bin_idx]}"

        components = {
            name: float(frac)
            for name, frac in (entry.get("components_frac") or {}).items()
        }
        out[channel] = {
            "region": region,
            "components_frac": components,
            "total_frac": float(entry.get("total_frac", 0.0)),
        }

    return out


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
