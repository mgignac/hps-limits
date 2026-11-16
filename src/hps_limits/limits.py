"""CLs upper limit computation for HPS search results."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyhf
import pyhf.infer
import pyhf.infer.intervals.upper_limits

from hps_limits.models import build_model

if TYPE_CHECKING:
    from hps_limits.io import BackgroundEntry, BackgroundEstimate, SearchInput, SignalEstimate


@dataclass
class LimitResult:
    """Result of a CLs upper limit calculation at a single mass point.

    Attributes
    ----------
    mass:
        A′ mass (MeV). ``None`` if not applicable.
    mu_upper_obs:
        Observed 95% CL upper limit on signal strength μ.
    mu_upper_exp:
        Expected (median) 95% CL upper limit on μ.
    mu_upper_exp_band:
        Expected limit band at [-2σ, -1σ, median, +1σ, +2σ].
    cls_obs:
        CLs value at μ=1 (cross-check / diagnostic).
    cls_exp:
        Expected CLs at μ=1.
    eps2_upper_obs:
        Observed 95% CL upper limit on ε².  Set by the direct ε² scan path.
    eps2_upper_exp:
        Expected (median) 95% CL upper limit on ε².
    eps2_upper_exp_band:
        Expected ε² limit band at [-2σ, -1σ, median, +1σ, +2σ].
    """

    mass: float | None
    mu_upper_obs: float
    mu_upper_exp: float
    mu_upper_exp_band: list[float]
    cls_obs: float | None = None
    cls_exp: float | None = None
    eps2_upper_obs: float | None = None
    eps2_upper_exp: float | None = None
    eps2_upper_exp_band: list[float] | None = None


# ---------------------------------------------------------------------------
# Core limit computation (shared by both APIs below)
# ---------------------------------------------------------------------------


def _run_limit(
    model: pyhf.Model,
    data: list[float],
    mass: float | None,
    cl_level: float,
    poi_upper: float,
) -> LimitResult:
    mu_obs, mu_exp_band = pyhf.infer.intervals.upper_limits.upper_limit(
        data,
        model,
        scan=np.linspace(0, poi_upper, 100),
        level=1 - cl_level,
        return_results=False,
    )

    cls_obs, cls_exp = pyhf.infer.hypotest(
        1.0, data, model, test_stat="qtilde", return_expected=True
    )

    return LimitResult(
        mass=mass,
        mu_upper_obs=float(mu_obs),
        mu_upper_exp=float(mu_exp_band[2]),  # median
        mu_upper_exp_band=[float(v) for v in mu_exp_band],
        cls_obs=float(cls_obs),
        cls_exp=float(cls_exp),
    )


# ---------------------------------------------------------------------------
# Simple programmatic API (single-channel, no JSON)
# ---------------------------------------------------------------------------


def compute_upper_limit(
    signal: float | list[float],
    background: float | list[float],
    background_uncertainty: float | list[float],
    observed: float | list[float] | None = None,
    mass: float | None = None,
    cl_level: float = 0.95,
    poi_upper: float = 10.0,
) -> LimitResult:
    """Compute the CLs upper limit on signal strength for one mass point.

    Convenience function for simple (uncorrelated background) models.
    For analyses with multiple regions or non-trivial systematics, use
    :func:`compute_upper_limit_from_input` instead.

    Parameters
    ----------
    signal:
        Expected signal yield(s).
    background:
        Expected background yield(s).
    background_uncertainty:
        Absolute 1σ background uncertainty for each channel.
    observed:
        Observed counts. Defaults to background (Asimov / expected-only).
    mass:
        A′ mass in MeV, stored in the result for bookkeeping.
    cl_level:
        Confidence level (default 0.95).
    poi_upper:
        Upper bound for the POI scan used by the root-finder.
    """
    model, default_data = build_model(signal, background, background_uncertainty)

    if observed is not None:
        obs = np.atleast_1d(np.asarray(observed, dtype=float)).tolist()
        data = obs + model.config.auxdata
    else:
        data = default_data

    return _run_limit(model, data, mass, cl_level, poi_upper)


def scan_mass_limits(
    masses: list[float],
    signals: list[float | list[float]],
    backgrounds: list[float | list[float]],
    background_uncertainties: list[float | list[float]],
    observeds: list[float | list[float] | None] | None = None,
    cl_level: float = 0.95,
    poi_upper: float = 10.0,
) -> list[LimitResult]:
    """Compute CLs upper limits across a range of A′ mass hypotheses.

    Convenience function for simple (uncorrelated background) models.
    For analyses with multiple regions or non-trivial systematics, use
    :func:`scan_mass_limits_from_input` instead.

    Parameters
    ----------
    masses:
        A′ mass values in MeV.
    signals:
        Signal yield(s) at each mass point.
    backgrounds:
        Background yield(s) at each mass point.
    background_uncertainties:
        Background uncertainty at each mass point.
    observeds:
        Observed counts at each mass point. ``None`` entries use background.
    cl_level:
        Confidence level (default 0.95).
    poi_upper:
        Upper POI bound for root-finding scan.
    """
    n = len(masses)
    if not (len(signals) == len(backgrounds) == len(background_uncertainties) == n):
        raise ValueError("All input lists must have the same length as masses.")

    if observeds is None:
        observeds = [None] * n

    return [
        compute_upper_limit(
            signal=signals[i],
            background=backgrounds[i],
            background_uncertainty=background_uncertainties[i],
            observed=observeds[i],
            mass=mass,
            cl_level=cl_level,
            poi_upper=poi_upper,
        )
        for i, mass in enumerate(masses)
    ]


# ---------------------------------------------------------------------------
# JSON-input API (multi-region, arbitrary systematics)
# ---------------------------------------------------------------------------


def compute_upper_limit_from_input(
    search: "SearchInput",
    signals: dict[str, float],
    mass: float | None = None,
    cl_level: float = 0.95,
    poi_upper: float = 10.0,
) -> LimitResult:
    """Compute the CLs upper limit for one mass point using a :class:`~hps_limits.io.SearchInput`.

    Parameters
    ----------
    search:
        Search configuration loaded from JSON (background, observed, systematics).
    signals:
        Expected signal yield in each region at this mass hypothesis,
        e.g. ``{"region_a": 5.0, "region_b": 3.0}``.
    mass:
        A′ mass in MeV, stored in the result for bookkeeping.
    cl_level:
        Confidence level (default 0.95).
    poi_upper:
        Upper bound for the POI scan used by the root-finder.
    """
    model, data = search.to_pyhf_model(signals)
    return _run_limit(model, data, mass, cl_level, poi_upper)


def scan_mass_limits_from_input(
    search: "SearchInput",
    signals_by_mass: dict[float, dict[str, float]],
    cl_level: float = 0.95,
    poi_upper: float = 10.0,
) -> list[LimitResult]:
    """Compute CLs upper limits across a range of A′ mass hypotheses.

    Parameters
    ----------
    search:
        Search configuration loaded from JSON.
    signals_by_mass:
        Mapping of A′ mass (MeV) to per-region signal yields, e.g.::

            {
                20.0: {"region_a": 8.0, "region_b": 5.0},
                25.0: {"region_a": 7.0, "region_b": 4.0},
            }

        Results are returned in the iteration order of this mapping.
    cl_level:
        Confidence level (default 0.95).
    poi_upper:
        Upper POI bound for root-finding scan.

    Returns
    -------
    list[LimitResult]
        One result per mass point, in the iteration order of ``signals_by_mass``.
    """
    return [
        compute_upper_limit_from_input(
            search=search,
            signals=signals,
            mass=mass,
            cl_level=cl_level,
            poi_upper=poi_upper,
        )
        for mass, signals in signals_by_mass.items()
    ]


# ---------------------------------------------------------------------------
# Direct ε² scan API (for CSV data with pre-computed Nsig vs ε² tables)
# ---------------------------------------------------------------------------


def _interp_eps2_limit(
    eps2_arr: np.ndarray,
    cls_arr: np.ndarray,
    threshold: float,
) -> float:
    """Interpolate on log(ε²) to find where CLs crosses ``threshold``.

    Assumes CLs is generally decreasing as ε² increases (higher signal →
    easier to exclude).  Returns ``nan`` if the threshold is never crossed
    within the scanned range.
    """
    below = cls_arr < threshold
    if not np.any(below):
        return float("nan")  # not excluded anywhere in the scan
    if not np.any(~below):
        return float(eps2_arr[0])  # excluded at every scanned point

    idx = int(np.argmax(below))  # first index below threshold
    if idx == 0:
        return float(eps2_arr[0])

    # Linear interpolation in log(ε²) between the straddling points.
    # cls_arr[idx-1] > threshold > cls_arr[idx], so x-values are ascending.
    log_limit = np.interp(
        threshold,
        [cls_arr[idx], cls_arr[idx - 1]],
        [np.log(eps2_arr[idx]), np.log(eps2_arr[idx - 1])],
    )
    return float(np.exp(log_limit))


def _find_eps2_crossings(
    eps2_arr: np.ndarray,
    cls_arr: np.ndarray,
    threshold: float,
) -> list[float]:
    """Find ALL ε² values where CLs crosses ``threshold``.

    Scans for sign changes in ``(cls_arr - threshold)`` and interpolates
    each crossing in log(ε²) space.  Returns an empty list if there are
    no crossings.

    Used for island-of-exclusion plots where CLs dips below the threshold
    in a finite ε² window (lower bound + upper bound per mass).
    """
    crossings = []
    diff = cls_arr - threshold

    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            # Sign change between i and i+1 — interpolate in log(ε²).
            # Sort so xp is increasing (required by np.interp); handles both
            # descending (island lower edge) and ascending (island upper edge).
            d0, d1 = diff[i], diff[i + 1]
            f0, f1 = np.log(eps2_arr[i]), np.log(eps2_arr[i + 1])
            if d0 > d1:
                d0, d1 = d1, d0
                f0, f1 = f1, f0
            log_cross = np.interp(0.0, [d0, d1], [f0, f1])
            crossings.append(float(np.exp(log_cross)))

    return crossings


@dataclass
class ClsScanResult:
    """Full CLs scan result at a single mass point.

    Attributes
    ----------
    mass:
        A′ mass (MeV).
    eps2_arr:
        ε² values used in the scan (filtered to valid signal).
    cls_obs_arr:
        Observed CLs at each ε².
    cls_exp_arr:
        Expected CLs band, shape ``(n_eps2, 5)`` for
        [−2σ, −1σ, median, +1σ, +2σ].
    limit:
        Extracted :class:`LimitResult` (eps2_upper_* fields populated).
    """

    mass: float
    eps2_arr: np.ndarray
    cls_obs_arr: np.ndarray
    cls_exp_arr: np.ndarray  # shape (n_eps2, 5)
    limit: LimitResult


def _run_cls_scan(
    eps2_arr: np.ndarray,
    nsig_arr: np.ndarray,
    nbkg: float,
    nbkg_unc: float,
    observed: float | None,
    mass: float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run pyhf hypotest(μ=1) at every valid (ε², Nsig) grid point.

    Filters out points where Nsig is below floating-point tiny (pyhf degenerates).

    Returns
    -------
    (eps2_filtered, cls_obs_arr, cls_exp_arr)
        eps2_filtered : shape (n,)
        cls_obs_arr   : shape (n,)
        cls_exp_arr   : shape (n, 5) — [−2σ, −1σ, median, +1σ, +2σ]
    """
    min_nsig = float(np.finfo(float).tiny)
    valid = nsig_arr > min_nsig
    n_skipped = int(np.sum(~valid))
    eps2_arr = eps2_arr[valid]
    nsig_arr = nsig_arr[valid]

    if verbose:
        mass_str = f"{mass:.0f} MeV" if mass is not None else "?"
        print(
            f"\n{'─'*72}\n"
            f"  Mass = {mass_str}   Nbkg = {nbkg:.3g}   Nbkg_unc = {nbkg_unc:.3g}"
            f"   obs = {'Asimov' if observed is None else observed}"
            f"   ({n_skipped} pts skipped, Nsig≤tiny)"
            f"\n  {'ε²':>12}  {'Nsig':>10}  {'CLs_obs':>10}  {'CLs_exp_med':>12}"
        )

    cls_obs_list = []
    cls_exp_rows = []

    for eps2, nsig in zip(eps2_arr, nsig_arr):
        model, default_data = build_model(
            signal=float(nsig),
            background=float(nbkg),
            background_uncertainty=float(nbkg_unc),
        )
        data = (
            [float(observed)] + model.config.auxdata
            if observed is not None
            else default_data
        )
        cls_obs_i, cls_exp_set_i = pyhf.infer.hypotest(
            1.0,
            data,
            model,
            test_stat="qtilde",
            return_expected_set=True,
        )
        cls_obs_val = float(cls_obs_i)
        cls_exp_row = [float(v) for v in cls_exp_set_i]

        if verbose:
            print(
                f"  {eps2:>12.4e}  {nsig:>10.4e}  {cls_obs_val:>10.4f}  {cls_exp_row[2]:>12.4f}"
            )

        cls_obs_list.append(cls_obs_val)
        cls_exp_rows.append(cls_exp_row)

    return (
        eps2_arr,
        np.array(cls_obs_list),
        np.array(cls_exp_rows),  # (n, 5)
    )


def _limit_result_from_cls_scan(
    eps2_arr: np.ndarray,
    cls_obs_arr: np.ndarray,
    cls_exp_arr: np.ndarray,
    mass: float | None,
    threshold: float,
) -> LimitResult:
    """Build a :class:`LimitResult` from pre-computed CLs arrays."""
    if len(eps2_arr) == 0:
        nan = float("nan")
        return LimitResult(
            mass=mass,
            mu_upper_obs=nan, mu_upper_exp=nan,
            mu_upper_exp_band=[nan] * 5,
            eps2_upper_obs=nan, eps2_upper_exp=nan,
            eps2_upper_exp_band=[nan] * 5,
        )

    eps2_obs = _interp_eps2_limit(eps2_arr, cls_obs_arr, threshold)
    eps2_exp_band = [
        _interp_eps2_limit(eps2_arr, cls_exp_arr[:, i], threshold)
        for i in range(5)
    ]
    eps2_exp = eps2_exp_band[2]
    ref = float(eps2_arr[-1])

    return LimitResult(
        mass=mass,
        mu_upper_obs=eps2_obs / ref,
        mu_upper_exp=eps2_exp / ref,
        mu_upper_exp_band=[e / ref for e in eps2_exp_band],
        eps2_upper_obs=eps2_obs,
        eps2_upper_exp=eps2_exp,
        eps2_upper_exp_band=eps2_exp_band,
    )


def compute_eps2_limit(
    eps2_values: np.ndarray | list,
    nsig_values: np.ndarray | list,
    nbkg: float,
    nbkg_unc: float | None = None,
    observed: float | None = None,
    mass: float | None = None,
    cl_level: float = 0.95,
) -> LimitResult:
    """Compute the ε² upper limit for one mass point by scanning CLs directly.

    For each ε² in ``eps2_values``, tests the signal hypothesis with the
    corresponding ``nsig`` yield against background-only (μ=1 test).
    Interpolates on log(ε²) to find the ε² where CLs = 1 − ``cl_level``.

    This approach is appropriate when Nsig is not simply proportional to ε²
    (e.g. displaced-vertex searches where the acceptance has a non-trivial
    ε² dependence).

    Parameters
    ----------
    eps2_values:
        ε² values (should be monotonically increasing).
    nsig_values:
        Expected signal yield at each ε².
    nbkg:
        Expected background yield.
    nbkg_unc:
        Absolute 1σ background uncertainty.  Defaults to ``sqrt(nbkg)``.
    observed:
        Observed count.  Defaults to ``nbkg`` (Asimov / expected-only).
    mass:
        A′ mass in MeV, stored in the result for bookkeeping.
    cl_level:
        Confidence level (default 0.95).

    Returns
    -------
    LimitResult
        ``eps2_upper_obs/exp/exp_band`` fields are populated.
        ``mu_upper_*`` fields are set to ``eps2_upper / eps2_values[-1]``
        (normalised to the largest scanned ε²) for compatibility with
        plotting functions that expect mu fields.
    """
    eps2_arr = np.asarray(eps2_values, dtype=float)
    nsig_arr = np.asarray(nsig_values, dtype=float)

    if nbkg_unc is None:
        nbkg_unc = float(np.sqrt(nbkg))

    eps2_arr, cls_obs_arr, cls_exp_arr = _run_cls_scan(
        eps2_arr, nsig_arr, nbkg, nbkg_unc, observed, mass=mass
    )
    return _limit_result_from_cls_scan(
        eps2_arr, cls_obs_arr, cls_exp_arr, mass, threshold=1.0 - cl_level
    )


def scan_eps2_limits(
    points: list[dict],
    nbkg_unc_frac: float | None = None,
    observed_by_mass: dict[float, float] | None = None,
    cl_level: float = 0.95,
) -> list[LimitResult]:
    """Compute ε² upper limits across all masses from a flat list of search points.

    Each point is one row from the CSV (one entry in the 2D (mass, ε²) grid).
    Points are grouped by mass; for each mass the full ε² scan is run and
    the limit is found by interpolation at CLs = 1 − ``cl_level``.

    Parameters
    ----------
    points:
        Output of :func:`~hps_limits.io.load_search_csv`.  Each entry is a
        dict with keys ``"mass_mev"``, ``"eps2"``, ``"nsig"``, ``"nbkg"``.
    nbkg_unc_frac:
        Fractional (relative) background uncertainty, e.g. ``0.10`` for 10 %.
        If ``None``, defaults to ``sqrt(Nbkg) / Nbkg`` (Poisson statistical).
    observed_by_mass:
        Observed event count at each mass (keyed by mass in MeV).  Defaults to
        ``nbkg`` (Asimov / expected-only limits).
    cl_level:
        Confidence level (default 0.95).

    Returns
    -------
    list[LimitResult]
        One result per mass, in ascending mass order.
    """
    # Group the flat list by mass, preserving per-point eps2/nsig/nbkg
    from collections import defaultdict

    by_mass: dict[float, list[dict]] = defaultdict(list)
    for pt in points:
        by_mass[pt["mass_mev"]].append(pt)

    results = []
    for mass_mev in sorted(by_mass):
        mass_points = sorted(by_mass[mass_mev], key=lambda p: p["eps2"])
        eps2_values = [p["eps2"] for p in mass_points]
        nsig_values = [p["nsig"] for p in mass_points]
        nbkg = mass_points[0]["nbkg"]
        nbkg_unc = (
            nbkg_unc_frac * nbkg
            if nbkg_unc_frac is not None
            else float(np.sqrt(nbkg))
        )
        observed = (
            observed_by_mass.get(mass_mev) if observed_by_mass is not None else None
        )
        results.append(
            compute_eps2_limit(
                eps2_values=eps2_values,
                nsig_values=nsig_values,
                nbkg=nbkg,
                nbkg_unc=nbkg_unc,
                observed=observed,
                mass=mass_mev,
                cl_level=cl_level,
            )
        )
    return results


def scan_cls_grid(
    points: list[dict],
    nbkg_unc_frac: float | None = None,
    observed_by_mass: dict[float, float] | None = None,
    cl_level: float = 0.95,
    verbose: bool = False,
) -> list[ClsScanResult]:
    """Run the full CLs scan grid and return per-mass raw CLs arrays.

    Like :func:`scan_eps2_limits` but returns the complete CLs(ε²) curves
    rather than just the extracted limit values.  Use this when you need
    island-of-exclusion plots (both lower and upper ε² boundaries per mass).

    Parameters
    ----------
    points:
        Output of :func:`~hps_limits.io.load_search_csv`.
    nbkg_unc_frac:
        Fractional background uncertainty.  ``None`` → Poisson (√Nbkg).
    observed_by_mass:
        Observed count per mass (MeV).  ``None`` → Asimov.
    cl_level:
        Confidence level (default 0.95).

    Returns
    -------
    list[ClsScanResult]
        One entry per mass in ascending mass order.  Each entry stores the
        full CLs arrays and the extracted :class:`LimitResult`.
    """
    from collections import defaultdict

    by_mass: dict[float, list[dict]] = defaultdict(list)
    for pt in points:
        by_mass[pt["mass_mev"]].append(pt)

    scan_results = []
    for mass_mev in sorted(by_mass):
        mass_points = sorted(by_mass[mass_mev], key=lambda p: p["eps2"])
        eps2_arr = np.asarray([p["eps2"] for p in mass_points], dtype=float)
        nsig_arr = np.asarray([p["nsig"] for p in mass_points], dtype=float)
        nbkg = mass_points[0]["nbkg"]
        nbkg_unc = (
            nbkg_unc_frac * nbkg
            if nbkg_unc_frac is not None
            else float(np.sqrt(nbkg))
        )
        observed = (
            observed_by_mass.get(mass_mev) if observed_by_mass is not None else None
        )

        eps2_filtered, cls_obs_arr, cls_exp_arr = _run_cls_scan(
            eps2_arr, nsig_arr, nbkg, nbkg_unc, observed, mass=mass_mev, verbose=verbose
        )
        limit = _limit_result_from_cls_scan(
            eps2_filtered, cls_obs_arr, cls_exp_arr, mass_mev, threshold=1.0 - cl_level
        )
        scan_results.append(
            ClsScanResult(
                mass=mass_mev,
                eps2_arr=eps2_filtered,
                cls_obs_arr=cls_obs_arr,
                cls_exp_arr=cls_exp_arr,
                limit=limit,
            )
        )
    return scan_results


# ---------------------------------------------------------------------------
# μ upper limit vs ε² scan (1D per mass)
# ---------------------------------------------------------------------------


@dataclass
class MuUpperScanResult:
    """Per-ε² signal-strength upper limit at a single mass point.

    At each ε² grid point, ``pyhf.infer.intervals.upper_limits.upper_limit``
    is called to find the μ value where CLs = 1 − cl_level.  A value of
    μ_upper < 1 means the nominal signal at that ε² is excluded.

    Attributes
    ----------
    mass:
        A′ mass (MeV).
    eps2_arr:
        ε² values (filtered to valid signal).
    mu_upper_obs_arr:
        Observed μ upper limit at each ε², shape ``(n,)``.
    mu_upper_exp_arr:
        Expected (median) μ upper limit at each ε², shape ``(n,)``.
    mu_upper_exp_band_arr:
        Full expected band, shape ``(n, 5)`` for [−2σ, −1σ, median, +1σ, +2σ].
    """

    mass: float | None
    eps2_arr: np.ndarray
    mu_upper_obs_arr: np.ndarray
    mu_upper_exp_arr: np.ndarray
    mu_upper_exp_band_arr: np.ndarray  # shape (n, 5)


def compute_mu_upper_scan(
    eps2_values: np.ndarray | list,
    nsig_values: np.ndarray | list,
    nbkg: float,
    nbkg_unc: float | None = None,
    observed: float | None = None,
    mass: float | None = None,
    cl_level: float = 0.95,
    mu_upper_cap: float = 10000.0,
    verbose: bool = False,
) -> MuUpperScanResult:
    """Compute the μ upper limit at every ε² grid point for one mass.

    For each ε², the signal model is built with ``signal = Nsig(ε²)`` and
    ``pyhf.infer.intervals.upper_limits.upper_limit`` finds the CLs = 0.05
    crossing in μ.  Points with Nsig ≤ float_tiny are skipped (returned as
    nan in the output arrays).

    To avoid spending time on clearly insensitive points, each ε² is
    pre-screened with a single ``hypotest(mu_upper_cap)`` call:

    * If ``CLs(mu_upper_cap) ≥ level`` → μ_upper ≥ cap; assign cap directly
      without calling ``upper_limit``.
    * Otherwise → compute the full limit and cap the result at ``mu_upper_cap``.

    Once the scan has found at least one point below the cap (the island) and
    then returns above it, the remaining ε² points are all assigned cap and
    the loop exits early.

    Parameters
    ----------
    eps2_values, nsig_values:
        ε² values and corresponding signal yields (monotonically increasing ε²).
    nbkg, nbkg_unc:
        Background yield and absolute 1σ uncertainty (defaults to √Nbkg).
    observed:
        Observed count.  ``None`` → Asimov.
    mass:
        A′ mass in MeV (stored for bookkeeping).
    cl_level:
        Confidence level (default 0.95).
    mu_upper_cap:
        Maximum μ to report.  Points above this are assigned the cap value and
        are used as early-stop triggers once the island has been seen.
        Default 10 000.
    verbose:
        Print a table of (ε², Nsig, μ_obs, μ_exp_median) as each point is computed.

    Returns
    -------
    MuUpperScanResult
    """
    eps2_arr = np.asarray(eps2_values, dtype=float)
    nsig_arr = np.asarray(nsig_values, dtype=float)

    if nbkg_unc is None:
        nbkg_unc = float(np.sqrt(nbkg))

    min_nsig = float(np.finfo(float).tiny)
    valid = nsig_arr > min_nsig
    eps2_arr = eps2_arr[valid]
    nsig_arr = nsig_arr[valid]

    level = 1.0 - cl_level

    if verbose:
        mass_str = f"{mass:.0f} MeV" if mass is not None else "?"
        print(
            f"\n{'─'*72}\n"
            f"  μ scan  Mass = {mass_str}   Nbkg = {nbkg:.3g}   Nbkg_unc = {nbkg_unc:.3g}"
            f"   obs = {'Asimov' if observed is None else observed}"
            f"   cap = {mu_upper_cap:.0f}"
            f"\n  {'ε²':>12}  {'Nsig':>10}  {'μ_obs':>10}  {'μ_exp_med':>12}"
        )

    mu_obs_list: list[float] = []
    mu_exp_band_rows: list[list[float]] = []
    seen_below_cap = False

    for i, (eps2, nsig) in enumerate(zip(eps2_arr, nsig_arr)):
        model, default_data = build_model(
            signal=float(nsig),
            background=float(nbkg),
            background_uncertainty=float(nbkg_unc),
        )
        data = (
            [float(observed)] + model.config.auxdata
            if observed is not None
            else default_data
        )
        # Wide par_bounds so the optimizer doesn't fail at large μ values.
        par_bounds = list(model.config.suggested_bounds())
        par_bounds[model.config.poi_index] = (0.0, float(mu_upper_cap) * 10)

        # --- Pre-screen ---------------------------------------------------
        # A single hypotest at mu_upper_cap is cheap.  If CLs(cap) ≥ level,
        # the true crossing is at μ > cap — assign cap and skip upper_limit.
        cls_at_cap = float(
            pyhf.infer.hypotest(
                float(mu_upper_cap),
                data,
                model,
                test_stat="qtilde",
                par_bounds=par_bounds,
                return_expected_set=False,
            )
        )
        above_cap = cls_at_cap >= level

        if above_cap:
            mu_obs_val = float(mu_upper_cap)
            mu_exp_row = [float(mu_upper_cap)] * 5
            if verbose:
                print(
                    f"  {eps2:>12.4e}  {nsig:>10.4e}  {'(cap)':>10}  {'(cap)':>12}"
                )
            mu_obs_list.append(mu_obs_val)
            mu_exp_band_rows.append(mu_exp_row)

            # Early exit: once the island has been found and we return above
            # cap, fill remaining points with cap and stop.
            if seen_below_cap:
                remaining = len(eps2_arr) - (i + 1)
                mu_obs_list.extend([float(mu_upper_cap)] * remaining)
                mu_exp_band_rows.extend([[float(mu_upper_cap)] * 5] * remaining)
                break
            continue

        # --- Full upper-limit computation ---------------------------------
        try:
            mu_obs, mu_exp_band = pyhf.infer.intervals.upper_limits.upper_limit(
                data,
                model,
                scan=None,
                level=level,
                test_stat="qtilde",
                par_bounds=par_bounds,
                return_results=False,
            )
        except (ValueError, RuntimeError):
            mu_obs = float(mu_upper_cap)
            mu_exp_band = [float(mu_upper_cap)] * 5

        mu_obs_val = min(float(mu_obs), float(mu_upper_cap))
        mu_exp_row = [min(float(v), float(mu_upper_cap)) for v in mu_exp_band]

        if mu_obs_val < mu_upper_cap:
            seen_below_cap = True

        if verbose:
            print(
                f"  {eps2:>12.4e}  {nsig:>10.4e}  {mu_obs_val:>10.4f}  {mu_exp_row[2]:>12.4f}"
            )

        mu_obs_list.append(mu_obs_val)
        mu_exp_band_rows.append(mu_exp_row)

    mu_exp_band_arr = np.array(mu_exp_band_rows)  # (n, 5)

    return MuUpperScanResult(
        mass=mass,
        eps2_arr=eps2_arr,
        mu_upper_obs_arr=np.array(mu_obs_list),
        mu_upper_exp_arr=mu_exp_band_arr[:, 2],  # median
        mu_upper_exp_band_arr=mu_exp_band_arr,
    )


def scan_mu_upper_grid(
    points: list[dict],
    nbkg_unc_frac: float | None = None,
    observed_by_mass: dict[float, float] | None = None,
    cl_level: float = 0.95,
    mu_upper_cap: float = 10000.0,
    verbose: bool = False,
) -> list[MuUpperScanResult]:
    """Compute per-ε² μ upper limits across all masses from a flat point list.

    Parameters
    ----------
    points:
        Output of :func:`~hps_limits.io.load_search_csv`.
    nbkg_unc_frac:
        Fractional background uncertainty.  ``None`` → Poisson (√Nbkg).
    observed_by_mass:
        Observed event count per mass (MeV).  ``None`` → Asimov.
    cl_level:
        Confidence level (default 0.95).
    mu_upper_cap:
        Maximum μ to report; also controls early stopping once the island has
        been found and sensitivity returns below this level.  Default 10 000.
    verbose:
        Print per-point output during the scan.

    Returns
    -------
    list[MuUpperScanResult]
        One entry per mass in ascending mass order.
    """
    from collections import defaultdict

    by_mass: dict[float, list[dict]] = defaultdict(list)
    for pt in points:
        by_mass[pt["mass_mev"]].append(pt)

    results = []
    for mass_mev in sorted(by_mass):
        mass_points = sorted(by_mass[mass_mev], key=lambda p: p["eps2"])
        eps2_values = [p["eps2"] for p in mass_points]
        nsig_values = [p["nsig"] for p in mass_points]
        nbkg = mass_points[0]["nbkg"]
        nbkg_unc = (
            nbkg_unc_frac * nbkg
            if nbkg_unc_frac is not None
            else float(np.sqrt(nbkg))
        )
        observed = (
            observed_by_mass.get(mass_mev) if observed_by_mass is not None else None
        )
        results.append(
            compute_mu_upper_scan(
                eps2_values=eps2_values,
                nsig_values=nsig_values,
                nbkg=nbkg,
                nbkg_unc=nbkg_unc,
                observed=observed,
                mass=mass_mev,
                cl_level=cl_level,
                mu_upper_cap=mu_upper_cap,
                verbose=verbose,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Estimate-based scans (BackgroundEstimate + SignalEstimate → multi-channel)
# ---------------------------------------------------------------------------


def _background_syst_modifiers(
    rname: str,
    background_systematics: dict | None,
    mode: str,
) -> list[dict]:
    """Build normsys modifiers for the background sample of channel *rname*.

    ``background_systematics`` is the mapping returned by
    :func:`~hps_limits.io.load_background_systematics_json`
    (``{channel: {"components_frac": {...}, "total_frac": float}}``).  Channels
    with no entry get no extra modifiers.

    ``mode`` controls the nuisance-parameter (NP) structure:

    ``"components"``
        One correlated NP per component (e.g. ``bkg_syst_sideband``,
        ``bkg_syst_control_region``), shared across channels by name.  Matches
        the JSON's documented combination (each component correlated across all
        region/bins, components independent of each other).
    ``"total"``
        A single correlated NP ``bkg_syst_total`` using ``total_frac`` (all
        channels' backgrounds move coherently).
    ``"total_uncorrelated"``
        A per-channel NP ``bkg_syst_<channel>`` using ``total_frac`` (each
        channel's background uncertainty independent).
    """
    if not background_systematics:
        return []
    entry = background_systematics.get(rname)
    if entry is None:
        return []

    mods: list[dict] = []
    if mode == "components":
        for comp_name, frac in entry.get("components_frac", {}).items():
            v = float(frac)
            if v <= 0.0:
                continue
            mods.append({
                "name": f"bkg_syst_{comp_name}",
                "type": "normsys",
                "data": {"hi": 1.0 + v, "lo": 1.0 - v},
            })
    elif mode == "total":
        v = float(entry.get("total_frac", 0.0))
        if v > 0.0:
            mods.append({
                "name": "bkg_syst_total",
                "type": "normsys",
                "data": {"hi": 1.0 + v, "lo": 1.0 - v},
            })
    elif mode == "total_uncorrelated":
        v = float(entry.get("total_frac", 0.0))
        if v > 0.0:
            mods.append({
                "name": f"bkg_syst_{rname}",
                "type": "normsys",
                "data": {"hi": 1.0 + v, "lo": 1.0 - v},
            })
    else:
        raise ValueError(
            f"Unknown background-systematics mode '{mode}'. Expected "
            "'components', 'total', or 'total_uncorrelated'."
        )
    return mods


def _pyhf_spec_from_entries(
    regions: list[str],
    nsig_by_region: dict[str, float],
    bg_by_region: dict[str, "BackgroundEntry"],
    nbkg_unc_frac: float | None = None,
    signal_systematics: list[dict] | None = None,
    background_systematics: dict | None = None,
    bkg_syst_mode: str = "components",
) -> dict:
    """Build a pyhf workspace spec from paired background and signal entries.

    Each region becomes one channel.  Background statistical uncertainty is
    modelled as an uncorrelated shapesys per region, using either the value
    from the :class:`~hps_limits.io.BackgroundEntry` or ``nbkg_unc_frac * nbkg``
    if an override fraction is supplied.

    Signal systematics are added as normsys modifiers with the same NP name
    across all channels (fully correlated).  Each entry in
    ``signal_systematics`` must have ``"name"`` (str) and ``"value"`` (float,
    relative 1σ).

    Background systematics (ABCD method uncertainties) are added as normsys
    modifiers on the background sample.  ``background_systematics`` is the
    mapping from :func:`~hps_limits.io.load_background_systematics_json` and
    ``bkg_syst_mode`` selects the correlation structure (see
    :func:`_background_syst_modifiers`).
    """
    channels = []
    observations = []

    for rname in regions:
        bg = bg_by_region[rname]
        nsig = max(float(nsig_by_region[rname]), 0.0)
        nbkg = max(float(bg.nbkg), 1e-10)
        if nbkg_unc_frac is not None:
            nbkg_err = float(nbkg_unc_frac) * nbkg
        else:
            nbkg_err = max(float(bg.nbkg_err), 1e-10)
        obs = float(bg.observed)

        sig_modifiers: list[dict] = [
            {"name": "mu", "type": "normfactor", "data": None}
        ]
        for syst in (signal_systematics or []):
            v = float(syst["value"])
            sig_modifiers.append({
                "name": syst["name"],
                "type": "normsys",
                "data": {"hi": 1.0 + v, "lo": 1.0 - v},
            })

        bkg_modifiers: list[dict] = [
            {
                "name": f"bkg_stat_{rname}",
                "type": "shapesys",
                "data": [nbkg_err],
            }
        ]
        bkg_modifiers.extend(
            _background_syst_modifiers(rname, background_systematics, bkg_syst_mode)
        )

        channels.append(
            {
                "name": rname,
                "samples": [
                    {
                        "name": "signal",
                        "data": [nsig],
                        "modifiers": sig_modifiers,
                    },
                    {
                        "name": "background",
                        "data": [nbkg],
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


def scan_cls_from_estimates(
    background: "BackgroundEstimate",
    signal: "SignalEstimate",
    select_regions: list[str] | None = None,
    nbkg_unc_frac: float | None = None,
    signal_systematics: list[dict] | None = None,
    background_systematics: dict | None = None,
    bkg_syst_mode: str = "components",
    cl_level: float = 0.95,
    verbose: bool = False,
) -> list[ClsScanResult]:
    """Run the CLs scan grid from paired background and signal JSON estimates.

    For each (mass, ε²) point in the signal estimate, a multi-channel pyhf
    model is built using the matching background entries (one channel per
    detector region).  ``hypotest(μ=1)`` is evaluated at every point to
    produce the CLs(ε²) curves needed for island-of-exclusion plots.

    Parameters
    ----------
    background:
        Background estimates loaded via :func:`~hps_limits.io.load_background_json`.
    signal:
        Signal yields loaded via :func:`~hps_limits.io.load_signal_json`.
    select_regions:
        If given, only these region names are included in the model.  Use to
        run a single hit category in isolation (e.g. ``["L1L1"]``).  ``None``
        (default) uses all regions common to background and signal.
    nbkg_unc_frac:
        Override the background uncertainty with this relative fraction.
        ``None`` (default) uses the ``nbkg_err`` stored in each
        :class:`~hps_limits.io.BackgroundEntry`.
    cl_level:
        Confidence level (default 0.95).
    verbose:
        Print a per-point table of (ε², Nsig_total, CLs_obs, CLs_exp_med).

    Returns
    -------
    list[ClsScanResult]
        One entry per common (background ∩ signal) mass, ascending order.
    """
    bg_by_mass = background.by_mass()
    sig_by_mass = signal.by_mass()
    common_masses = sorted(set(bg_by_mass) & set(sig_by_mass))
    threshold = 1.0 - cl_level

    results = []
    for mass_mev in common_masses:
        bg_entries = bg_by_mass[mass_mev]
        sig_entries = sig_by_mass[mass_mev]

        bg_by_region = {e.region: e for e in bg_entries}
        sig_regions = {e.region for e in sig_entries}
        regions = [r for r in background.regions if r in bg_by_region and r in sig_regions]
        if select_regions is not None:
            regions = [r for r in regions if r in select_regions]
        if not regions:
            continue

        if verbose and any(bg_by_region[r].nbkg <= 0 for r in regions):
            zero = [r for r in regions if bg_by_region[r].nbkg <= 0]
            print(f"  mass={mass_mev:.0f} MeV: zero background in {zero}, clamping to 1e-10")

        sig_by_region_eps2: dict[tuple[str, float], float] = {
            (e.region, e.eps2): e.nsig for e in sig_entries
        }
        eps2_arr = np.array(sorted({e.eps2 for e in sig_entries}))

        if verbose:
            print(
                f"\n{'─'*72}\n"
                f"  CLs scan  mass={mass_mev:.0f} MeV  regions={regions}"
                f"\n  {'ε²':>12}  {'Nsig_tot':>10}  {'CLs_obs':>10}  {'CLs_exp_med':>12}"
            )

        cls_obs_list: list[float] = []
        cls_exp_rows: list[list[float]] = []
        valid_eps2: list[float] = []

        for eps2 in eps2_arr:
            nsig_by_region = {
                r: sig_by_region_eps2.get((r, eps2), 0.0) for r in regions
            }
            total_nsig = sum(nsig_by_region.values())
            if total_nsig <= float(np.finfo(float).tiny):
                continue

            spec = _pyhf_spec_from_entries(
                regions, nsig_by_region, bg_by_region, nbkg_unc_frac,
                signal_systematics, background_systematics, bkg_syst_mode,
            )
            ws = pyhf.Workspace(spec)
            model = ws.model()
            data = ws.data(model)

            try:
                cls_obs_i, cls_exp_set_i = pyhf.infer.hypotest(
                    1.0, data, model, test_stat="qtilde", return_expected_set=True
                )
                cls_obs_val = float(cls_obs_i)
                cls_exp_row = [float(v) for v in cls_exp_set_i]
            except Exception as exc:
                if verbose:
                    print(
                        f"  {eps2:>12.4e}  {total_nsig:>10.4e}"
                        f"  [optimizer failed: {type(exc).__name__}]"
                    )
                continue

            # Replace NaN (0/0 in CLs) with 1.0 — conservatively not excluded
            if np.isnan(cls_obs_val):
                cls_obs_val = 1.0
            cls_exp_row = [1.0 if np.isnan(v) else v for v in cls_exp_row]

            if verbose:
                print(
                    f"  {eps2:>12.4e}  {total_nsig:>10.4e}"
                    f"  {cls_obs_val:>10.4f}  {cls_exp_row[2]:>12.4f}"
                )

            cls_obs_list.append(cls_obs_val)
            cls_exp_rows.append(cls_exp_row)
            valid_eps2.append(eps2)

        if not valid_eps2:
            continue

        eps2_valid = np.array(valid_eps2)
        cls_obs_arr = np.array(cls_obs_list)
        cls_exp_arr = np.array(cls_exp_rows)
        limit = _limit_result_from_cls_scan(
            eps2_valid, cls_obs_arr, cls_exp_arr, mass_mev, threshold
        )
        results.append(
            ClsScanResult(
                mass=mass_mev,
                eps2_arr=eps2_valid,
                cls_obs_arr=cls_obs_arr,
                cls_exp_arr=cls_exp_arr,
                limit=limit,
            )
        )

    return results


def _mu_upper_from_cls_scan(
    data,
    model,
    level: float,
    par_bounds: list,
    mu_upper_cap: float,
    n_grid: int = 64,
) -> tuple[float, list[float]]:
    """Robust fallback μ-upper-limit via a direct CLs scan + interpolation.

    Used when pyhf's toms748 root-finder fails: it probes μ → 0, where the
    asymptotic ``qtilde`` test statistic is 0/0 → NaN, raising
    ``ValueError: f(0.0) -> nan`` (and the caller would otherwise pin the point
    to the cap, producing spurious spikes).  ``hypotest`` itself is finite at
    finite μ, so here CLs (observed + the 5-member expected band) is evaluated
    on a log-spaced μ grid, non-finite nodes are dropped, and the CLs = *level*
    crossing is located per series by log-log interpolation between the
    bracketing grid points.

    Returns ``(mu_obs, mu_exp_band)`` with the same 5-element band ordering as
    :func:`pyhf.infer.intervals.upper_limits.upper_limit`.
    """
    from scipy.optimize import brentq

    # cap·1e-6 … cap (e.g. 1e-2 … 1e4 for cap=1e4): low enough to bracket even
    # sub-1 limits, high enough to reach the cap.
    lo = max(float(mu_upper_cap) * 1e-6, 1e-6)
    mus = np.logspace(np.log10(lo), np.log10(float(mu_upper_cap)), n_grid)

    _cache: dict[float, list[float]] = {}

    def _clsvec(mu: float) -> list[float]:
        """[CLs_obs, CLs_-2σ, -1σ, med, +1σ, +2σ] at μ; NaNs if hypotest fails."""
        key = float(mu)
        if key in _cache:
            return _cache[key]
        try:
            cls_obs, cls_exp = pyhf.infer.hypotest(
                key, data, model, test_stat="qtilde",
                par_bounds=par_bounds, return_expected_set=True,
            )
            vals = [float(cls_obs)] + [float(v) for v in cls_exp]
        except Exception:
            vals = [float("nan")] * 6
        _cache[key] = vals
        return vals

    grid = [(float(mu), _clsvec(float(mu))) for mu in mus]

    def _crossing(s: int) -> float:
        # CLs is monotonically decreasing in μ; the first finite node at/below
        # *level* brackets the limit, then refine the root between the two
        # bracketing nodes (where hypotest is finite) to match the toms748 path.
        pts = [(mu, v[s]) for mu, v in grid if np.isfinite(v[s])]
        if len(pts) < 2:
            return float(mu_upper_cap)
        for k in range(len(pts)):
            if pts[k][1] <= level:
                if k == 0:
                    return pts[0][0]          # limit below grid (rare); conservative
                m0, c0 = pts[k - 1]
                m1, c1 = pts[k]
                f = lambda mu: _clsvec(mu)[s] - level
                try:
                    return float(brentq(f, m0, m1, rtol=1e-4, maxiter=80))
                except Exception:
                    pass
                # log-log interpolation fallback if the root-find misbehaves
                if c0 > 0.0 and c1 > 0.0 and c0 != c1:
                    lm = (np.log(m0)
                          + (np.log(level) - np.log(c0))
                          * (np.log(m1) - np.log(m0)) / (np.log(c1) - np.log(c0)))
                    return float(np.exp(lm))
                t = (level - c0) / (c1 - c0) if c1 != c0 else 0.0
                return m0 + t * (m1 - m0)
        return float(mu_upper_cap)            # never crossed → at/above cap

    crossings = [_crossing(s) for s in range(6)]
    return crossings[0], crossings[1:]


def scan_mu_upper_from_estimates(
    background: "BackgroundEstimate",
    signal: "SignalEstimate",
    select_regions: list[str] | None = None,
    nbkg_unc_frac: float | None = None,
    signal_systematics: list[dict] | None = None,
    background_systematics: dict | None = None,
    bkg_syst_mode: str = "components",
    cl_level: float = 0.95,
    mu_upper_cap: float = 10000.0,
    verbose: bool = False,
) -> list[MuUpperScanResult]:
    """Compute per-ε² μ upper limits from paired background and signal estimates.

    Mirrors :func:`scan_mu_upper_grid` but accepts
    :class:`~hps_limits.io.BackgroundEstimate` and
    :class:`~hps_limits.io.SignalEstimate` objects directly.  A multi-channel
    pyhf model is built per (mass, ε²) and
    ``pyhf.infer.intervals.upper_limits.upper_limit`` (toms748 root-finding)
    finds the μ where CLs = 1 − ``cl_level``.

    Parameters
    ----------
    background:
        Background estimates loaded via :func:`~hps_limits.io.load_background_json`.
    signal:
        Signal yields loaded via :func:`~hps_limits.io.load_signal_json`.
    select_regions:
        If given, only these region names are included in the model.  Use to
        run a single hit category in isolation (e.g. ``["L1L1"]``).  ``None``
        (default) uses all regions common to background and signal.
    nbkg_unc_frac:
        Override background uncertainty fraction.  ``None`` uses stored ``nbkg_err``.
    cl_level:
        Confidence level (default 0.95).
    mu_upper_cap:
        Cap on reported μ_upper; also triggers early stopping after the island.
    verbose:
        Print per-point progress table.

    Returns
    -------
    list[MuUpperScanResult]
        One entry per common mass, ascending order.
    """
    bg_by_mass = background.by_mass()
    sig_by_mass = signal.by_mass()
    common_masses = sorted(set(bg_by_mass) & set(sig_by_mass))
    level = 1.0 - cl_level

    results = []
    for mass_mev in common_masses:
        bg_entries = bg_by_mass[mass_mev]
        sig_entries = sig_by_mass[mass_mev]

        bg_by_region = {e.region: e for e in bg_entries}
        sig_regions = {e.region for e in sig_entries}
        regions = [r for r in background.regions if r in bg_by_region and r in sig_regions]
        if select_regions is not None:
            regions = [r for r in regions if r in select_regions]
        if not regions:
            continue

        if verbose and any(bg_by_region[r].nbkg <= 0 for r in regions):
            zero = [r for r in regions if bg_by_region[r].nbkg <= 0]
            print(f"  mass={mass_mev:.0f} MeV: zero background in {zero}, clamping to 1e-10")

        sig_by_region_eps2: dict[tuple[str, float], float] = {
            (e.region, e.eps2): e.nsig for e in sig_entries
        }
        eps2_arr = np.array(sorted({e.eps2 for e in sig_entries}))

        if verbose:
            print(
                f"\n{'─'*72}\n"
                f"  μ scan  mass={mass_mev:.0f} MeV  regions={regions}  cap={mu_upper_cap:.0f}"
                f"\n  {'ε²':>12}  {'Nsig_tot':>10}  {'μ_obs':>10}  {'μ_exp_med':>12}"
            )

        mu_obs_list: list[float] = []
        mu_exp_band_rows: list[list[float]] = []
        seen_below_cap = False
        n_fallback = 0
        valid_eps2: list[float] = []

        for i, eps2 in enumerate(eps2_arr):
            nsig_by_region = {
                r: sig_by_region_eps2.get((r, eps2), 0.0) for r in regions
            }
            total_nsig = sum(nsig_by_region.values())
            if total_nsig <= float(np.finfo(float).tiny):
                continue

            spec = _pyhf_spec_from_entries(
                regions, nsig_by_region, bg_by_region, nbkg_unc_frac,
                signal_systematics, background_systematics, bkg_syst_mode,
            )
            ws = pyhf.Workspace(spec)
            model = ws.model()
            data = ws.data(model)

            par_bounds = list(model.config.suggested_bounds())
            par_bounds[model.config.poi_index] = (0.0, float(mu_upper_cap) * 10)

            # Pre-screen: single hypotest at the cap value
            try:
                cls_at_cap = float(
                    pyhf.infer.hypotest(
                        float(mu_upper_cap),
                        data,
                        model,
                        test_stat="qtilde",
                        par_bounds=par_bounds,
                        return_expected_set=False,
                    )
                )
                if np.isnan(cls_at_cap):
                    cls_at_cap = 1.0
            except Exception:
                cls_at_cap = 1.0  # treat optimizer failure as above-cap
            above_cap = cls_at_cap >= level

            if above_cap:
                mu_obs_val = float(mu_upper_cap)
                mu_exp_row = [float(mu_upper_cap)] * 5
                if verbose:
                    print(
                        f"  {eps2:>12.4e}  {total_nsig:>10.4e}"
                        f"  {'(cap)':>10}  {'(cap)':>12}"
                    )
                valid_eps2.append(eps2)
                mu_obs_list.append(mu_obs_val)
                mu_exp_band_rows.append(mu_exp_row)

                if seen_below_cap:
                    remaining_eps2 = [e for e in eps2_arr[i + 1:]]
                    valid_eps2.extend(remaining_eps2)
                    mu_obs_list.extend([float(mu_upper_cap)] * len(remaining_eps2))
                    mu_exp_band_rows.extend(
                        [[float(mu_upper_cap)] * 5] * len(remaining_eps2)
                    )
                    break
                continue

            # Full upper-limit computation: fast asymptotic root-finder, with a
            # robust CLs-scan fallback when it fails or returns a spurious value.
            #
            # We call ``toms748_scan`` directly (not ``upper_limit``) and pass
            # ``level`` explicitly: pyhf 0.7.6's ``upper_limit(scan=None)`` drops
            # the ``level`` argument and always solves CLs = 0.05, so going
            # through it would compute a 95% CL limit regardless of cl_level.
            #
            # The fallback handles the points where toms748 probes μ → 0, where
            # asymptotic qtilde is 0/0 → NaN (``f(0.0) -> nan``).  The pre-screen
            # above already guarantees the true limit is below the cap here, so
            # an exception, a non-finite result, or a value pinned at the cap is
            # spurious and triggers the fallback.
            used_fallback = False
            poi_idx = model.config.poi_index
            b_lo, b_hi = model.config.suggested_bounds()[poi_idx]
            try:
                mu_obs, mu_exp_band = pyhf.infer.intervals.upper_limits.toms748_scan(
                    data,
                    model,
                    b_lo,
                    b_hi,
                    level=level,
                    test_stat="qtilde",
                    par_bounds=par_bounds,
                )
                mu_obs = float(mu_obs)
                mu_exp_band = [float(v) for v in mu_exp_band]
                if (not np.isfinite(mu_obs)
                        or not all(np.isfinite(v) for v in mu_exp_band)
                        or mu_obs >= float(mu_upper_cap)):
                    used_fallback = True
            except Exception:
                used_fallback = True

            if used_fallback:
                mu_obs, mu_exp_band = _mu_upper_from_cls_scan(
                    data, model, level, par_bounds, mu_upper_cap,
                )
                n_fallback += 1

            mu_obs_val = min(float(mu_obs), float(mu_upper_cap))
            mu_exp_row = [min(float(v), float(mu_upper_cap)) for v in mu_exp_band]

            if mu_obs_val < mu_upper_cap:
                seen_below_cap = True

            if verbose:
                print(
                    f"  {eps2:>12.4e}  {total_nsig:>10.4e}"
                    f"  {mu_obs_val:>10.4f}  {mu_exp_row[2]:>12.4f}"
                    f"{'  [fallback]' if used_fallback else ''}"
                )

            valid_eps2.append(eps2)
            mu_obs_list.append(mu_obs_val)
            mu_exp_band_rows.append(mu_exp_row)

            # pyhf Workspace/model objects contain reference cycles (config ↔
            # model), so they are reclaimed only by the cyclic collector, not
            # by refcounting.  Over a full ε² × mass grid these accumulate and
            # can exhaust memory; collect periodically to bound the footprint.
            if i % 25 == 0:
                gc.collect()

        gc.collect()

        if verbose and n_fallback:
            print(f"  [robust CLs-scan fallback used at {n_fallback} ε² point(s)]")

        if not valid_eps2:
            continue

        mu_exp_band_arr = np.array(mu_exp_band_rows)
        results.append(
            MuUpperScanResult(
                mass=mass_mev,
                eps2_arr=np.array(valid_eps2),
                mu_upper_obs_arr=np.array(mu_obs_list),
                mu_upper_exp_arr=mu_exp_band_arr[:, 2],
                mu_upper_exp_band_arr=mu_exp_band_arr,
            )
        )

    return results


def broaden_mu_band_with_systematic(
    nominal: list["MuUpperScanResult"],
    up: list["MuUpperScanResult"],
    down: list["MuUpperScanResult"],
) -> None:
    """Broaden the expected μ-upper band of *nominal* by a background systematic.

    ``up`` / ``down`` are the **same** μ scan recomputed with every channel's
    background scaled by ``(1 ± σ_total)`` (the per-channel total ABCD
    systematic).  The shift of the *median* expected limit between nominal and
    the ±1σ-shifted backgrounds is taken as the 1σ systematic uncertainty on the
    expected limit, and combined in quadrature with the existing statistical
    band, edge by edge.  The median (index 2) is left unchanged; only the
    ±1σ (indices 1, 3) and ±2σ (indices 0, 4) edges move outward.

    Modifies ``nominal`` in place.  The observed limit is untouched — the
    systematic is applied to the expected band only.
    """
    up_by_mass = {round(r.mass, 6): r for r in up}
    down_by_mass = {round(r.mass, 6): r for r in down}

    # (band index, number of sigmas the systematic contributes, sign)
    edges = ((0, 2.0, -1.0), (1, 1.0, -1.0), (3, 1.0, +1.0), (4, 2.0, +1.0))

    for res in nominal:
        u = up_by_mass.get(round(res.mass, 6))
        d = down_by_mass.get(round(res.mass, 6))
        if u is None or d is None:
            continue

        u_med = {round(float(e), 12): m for e, m in zip(u.eps2_arr, u.mu_upper_exp_arr)}
        d_med = {round(float(e), 12): m for e, m in zip(d.eps2_arr, d.mu_upper_exp_arr)}

        band = res.mu_upper_exp_band_arr.astype(float).copy()
        for i, eps2 in enumerate(res.eps2_arr):
            key = round(float(eps2), 12)
            med = float(res.mu_upper_exp_arr[i])
            mu_u = u_med.get(key)
            mu_d = d_med.get(key)
            if mu_u is None or mu_d is None or not np.isfinite(med):
                continue
            delta = max(abs(mu_u - med), abs(mu_d - med))
            if not np.isfinite(delta) or delta <= 0.0:
                continue
            for q, nsig, sign in edges:
                stat_dev = float(band[i, q]) - med
                if not np.isfinite(stat_dev):
                    continue
                band[i, q] = med + sign * float(np.hypot(stat_dev, nsig * delta))
        res.mu_upper_exp_band_arr = band
