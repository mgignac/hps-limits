"""Plotting utilities for HPS limit results.

Requires matplotlib (install with ``pip install hps-limits[plotting]``).

The kinematic-mixing conversion
---------------------------------
Signal yield scales as S(ε²) = S_ref × (ε² / ε²_ref), so the upper limit
on signal strength μ maps directly to an upper limit on ε²::

    ε²_limit = μ_limit × ε²_ref

All plot functions that display ε² on an axis accept either a single
``eps2_ref`` (same reference for every mass) or a per-mass dict
``eps2_by_mass`` (for cases where the MC reference coupling varies).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    from hps_limits.io import SearchInput
    from hps_limits.limits import ClsScanResult, LimitResult, MuUpperScanResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_ax(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    return ax


def _eps2_scale(
    results: list["LimitResult"],
    eps2_ref: float | None,
    eps2_by_mass: dict[float, float] | None,
) -> list[float] | None:
    """Return per-mass ε²_ref scaling factors, or None if not provided."""
    if eps2_by_mass is not None:
        return [eps2_by_mass[r.mass] for r in results]
    if eps2_ref is not None:
        return [eps2_ref] * len(results)
    return None


# ---------------------------------------------------------------------------
# 2D exclusion plot
# ---------------------------------------------------------------------------


def plot_exclusion(
    results: list["LimitResult"],
    eps2_ref: float | None = None,
    eps2_by_mass: dict[float, float] | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    show_band: bool = True,
    obs_style: dict | None = None,
    exp_style: dict | None = None,
) -> "matplotlib.axes.Axes":
    """2D exclusion plot: limit vs A′ mass, with ±1σ / ±2σ expected band.

    Parameters
    ----------
    results:
        Output of :func:`~hps_limits.scan_mass_limits_from_input` (or the
        simple variant).  Each entry must have a non-``None`` ``mass``.
    eps2_ref:
        Reference ε² used when computing MC signal yields (same for all
        masses).  If provided the y-axis shows ε²_limit; otherwise it shows
        the dimensionless μ upper limit.
    eps2_by_mass:
        Per-mass reference ε², keyed by the same float values used in
        ``results[i].mass``.  Takes precedence over ``eps2_ref``.
    ax:
        Existing Axes to draw on.  A new figure is created if ``None``.
    show_band:
        Whether to draw the ±1σ / ±2σ expected-limit bands.
    obs_style, exp_style:
        Extra keyword arguments forwarded to ``ax.plot`` for the observed
        and expected-median lines respectively.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_ax(ax)
    masses = [r.mass for r in results]

    # Prefer eps2 fields stored directly on LimitResult (direct-scan path),
    # then fall back to scaling mu limits by eps2_ref / eps2_by_mass.
    use_direct = all(r.eps2_upper_obs is not None for r in results)

    if use_direct:
        obs  = [r.eps2_upper_obs for r in results]
        exp  = [r.eps2_upper_exp for r in results]
        band = [[r.eps2_upper_exp_band[j] for r in results] for j in range(5)]
        using_eps2 = True
    else:
        scales = _eps2_scale(results, eps2_ref, eps2_by_mass)

        def _scale(mu_val, i):
            return mu_val * scales[i] if scales is not None else mu_val

        obs  = [_scale(r.mu_upper_obs, i) for i, r in enumerate(results)]
        exp  = [_scale(r.mu_upper_exp, i) for i, r in enumerate(results)]
        band = [
            [_scale(r.mu_upper_exp_band[j], i) for i, r in enumerate(results)]
            for j in range(5)
        ]
        using_eps2 = scales is not None

    if show_band:
        ax.fill_between(
            masses,
            band[0],
            band[4],
            color="gold",
            alpha=0.6,
            label=r"Exp. $\pm 2\sigma$",
            zorder=1,
        )
        ax.fill_between(
            masses,
            band[1],
            band[3],
            color="limegreen",
            alpha=0.8,
            label=r"Exp. $\pm 1\sigma$",
            zorder=2,
        )

    _exp_style = dict(color="black", linestyle="--", linewidth=1.5,
                      label="Expected 95% CL", zorder=3)
    if exp_style:
        _exp_style.update(exp_style)
    ax.plot(masses, exp, **_exp_style)

    _obs_style = dict(color="black", linestyle="-", linewidth=2.0,
                      label="Observed 95% CL", zorder=4)
    if obs_style:
        _obs_style.update(obs_style)
    ax.plot(masses, obs, **_obs_style)

    ax.set_xlabel(r"$m_{A'}$ (MeV)")
    if using_eps2:
        ax.set_ylabel(r"$\varepsilon^2$ upper limit (95% CL)")
    else:
        ax.set_ylabel(r"$\mu$ upper limit (95% CL)")
    ax.set_yscale("log")
    ax.legend()

    return ax


# ---------------------------------------------------------------------------
# Island-of-exclusion plot (displaced vertex searches)
# ---------------------------------------------------------------------------


def plot_island_exclusion(
    scan_results: list["ClsScanResult"],
    ax: "matplotlib.axes.Axes | None" = None,
    cl_level: float = 0.95,
    show_band: bool = True,
    obs_style: dict | None = None,
    exp_style: dict | None = None,
    ylim: tuple[float, float] | None = None,
    eps_axis: bool = False,
) -> "matplotlib.axes.Axes":
    """Plot the island of exclusion for a displaced-vertex search.

    For each mass, finds all ε² values where CLs = 1 − ``cl_level`` crosses
    (the threshold).  When there are two crossings (lower and upper ε²
    boundaries), the interval between them is the excluded island.  The
    island is filled and the boundary contours are drawn.

    Parameters
    ----------
    scan_results:
        Output of :func:`~hps_limits.limits.scan_cls_grid`.
    ax:
        Existing Axes to draw on.  A new figure is created if ``None``.
    cl_level:
        Confidence level (default 0.95).
    show_band:
        Whether to shade the ±1σ / ±2σ expected bands around the island
        boundary.
    obs_style, exp_style:
        Extra keyword arguments forwarded to ``ax.plot`` for the observed
        and expected-median boundary lines.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from hps_limits.limits import _find_eps2_crossings

    ax = _get_ax(ax)
    threshold = 1.0 - cl_level

    masses = []
    obs_lower, obs_upper = [], []
    exp_lower, exp_upper = [], []
    # per-band lower/upper:  band_lo[i], band_hi[i] for i in range(5)
    band_lo = [[] for _ in range(5)]
    band_hi = [[] for _ in range(5)]

    for sr in scan_results:
        if len(sr.eps2_arr) == 0:
            continue

        obs_cross = _find_eps2_crossings(sr.eps2_arr, sr.cls_obs_arr, threshold)
        exp_cross = _find_eps2_crossings(sr.eps2_arr, sr.cls_exp_arr[:, 2], threshold)
        band_crossings = [
            _find_eps2_crossings(sr.eps2_arr, sr.cls_exp_arr[:, i], threshold)
            for i in range(5)
        ]

        # Include this mass if *any* curve (obs or any expected band) has an island
        any_island = len(obs_cross) >= 2 or any(
            len(bc) >= 2 for bc in band_crossings
        )
        if not any_island:
            continue

        masses.append(sr.mass)
        obs_lower.append(obs_cross[0] if len(obs_cross) >= 2 else float("nan"))
        obs_upper.append(obs_cross[-1] if len(obs_cross) >= 2 else float("nan"))
        exp_lower.append(exp_cross[0] if len(exp_cross) >= 2 else float("nan"))
        exp_upper.append(exp_cross[-1] if len(exp_cross) >= 2 else float("nan"))

        for i, bc in enumerate(band_crossings):
            if len(bc) >= 2:
                band_lo[i].append(bc[0])
                band_hi[i].append(bc[-1])
            else:
                band_lo[i].append(float("nan"))
                band_hi[i].append(float("nan"))

    _coupling = np.sqrt if eps_axis else lambda x: x
    cl_pct = int(round(cl_level * 100))

    if not masses:
        ax.set_xlabel(r"$m_{A'}$ (MeV)")
        ax.set_ylabel((rf"$\varepsilon$ ({cl_pct}% CL island)" if eps_axis
                       else rf"$\varepsilon^2$ ({cl_pct}% CL island)"))
        ax.set_yscale("log")
        return ax

    masses_arr = np.array(masses)

    def _masked(lst):
        """Convert a list (possibly containing nan) to a masked array."""
        a = np.array(lst, dtype=float)
        return np.ma.masked_invalid(_coupling(a))

    # Convert all boundary lists to masked arrays so nan gaps render cleanly
    obs_lo_a  = _masked(obs_lower)
    obs_hi_a  = _masked(obs_upper)
    exp_lo_a  = _masked(exp_lower)
    exp_hi_a  = _masked(exp_upper)
    blo = [_masked(band_lo[i]) for i in range(5)]
    bhi = [_masked(band_hi[i]) for i in range(5)]

    if show_band:
        # ±2σ band on the lower boundary  (blo[0]=−2σ crossing < blo[4]=+2σ crossing)
        ax.fill_between(
            masses_arr, blo[0], blo[4],
            color="gold", alpha=0.6, label=r"Exp. $\pm 2\sigma$", zorder=1,
        )
        # ±2σ band on the upper boundary  (bhi[4]=+2σ crossing < bhi[0]=−2σ crossing)
        ax.fill_between(
            masses_arr, bhi[4], bhi[0],
            color="gold", alpha=0.6, zorder=1,
        )
        # ±1σ band on the lower boundary
        ax.fill_between(
            masses_arr, blo[1], blo[3],
            color="limegreen", alpha=0.8, label=r"Exp. $\pm 1\sigma$", zorder=2,
        )
        # ±1σ band on the upper boundary
        ax.fill_between(
            masses_arr, bhi[3], bhi[1],
            color="limegreen", alpha=0.8, zorder=2,
        )

    _exp_style = dict(color="black", linestyle="--", linewidth=1.5,
                      label=f"Expected {cl_pct}% CL", zorder=3)
    if exp_style:
        _exp_style.update(exp_style)

    # Draw both boundaries of the expected island
    ax.plot(masses_arr, exp_lo_a, **_exp_style)
    _exp_upper_style = dict(_exp_style)
    _exp_upper_style.pop("label", None)
    ax.plot(masses_arr, exp_hi_a, **_exp_upper_style)

    _obs_style = dict(color="black", linestyle="-", linewidth=2.0,
                      label=f"Observed {cl_pct}% CL", zorder=4)
    if obs_style:
        _obs_style.update(obs_style)

    ax.plot(masses_arr, obs_lo_a, **_obs_style)
    _obs_upper_style = dict(_obs_style)
    _obs_upper_style.pop("label", None)
    ax.plot(masses_arr, obs_hi_a, **_obs_upper_style)

    # Shade the observed excluded island
    ax.fill_between(
        masses_arr, obs_lo_a, obs_hi_a,
        color="royalblue", alpha=0.25, label="Excluded (obs.)", zorder=0,
    )

    ax.set_xlabel(r"$m_{A'}$ (MeV)")
    ax.set_ylabel((rf"$\varepsilon$ ({cl_pct}% CL island)" if eps_axis
                   else rf"$\varepsilon^2$ ({cl_pct}% CL island)"))
    ax.set_yscale("log")

    if ylim is not None:
        ax.set_ylim(_coupling(ylim[0]), _coupling(ylim[1]))
    else:
        all_eps2 = np.concatenate([sr.eps2_arr for sr in scan_results if len(sr.eps2_arr) > 0])
        if len(all_eps2) > 0:
            ax.set_ylim(_coupling(all_eps2.min()), _coupling(all_eps2.max()))

    ax.legend()

    return ax


# ---------------------------------------------------------------------------
# 1D CLs scan
# ---------------------------------------------------------------------------


def compute_cls_scan(
    search: "SearchInput",
    signals: dict[str, float],
    poi_range: tuple[float, float] = (0.0, 10.0),
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Compute CLs as a function of μ at a fixed mass point.

    Parameters
    ----------
    search:
        Search configuration (background, observed, systematics).
    signals:
        Expected signal yield per region at the reference ε².
    poi_range:
        (min, max) range of μ values to scan.
    n_points:
        Number of μ values in the scan.

    Returns
    -------
    mu_values : np.ndarray, shape (n_points,)
    cls_obs   : np.ndarray, shape (n_points,)
    cls_exp_band : list of 5 np.ndarray, shape (n_points,)
        Bands in order [-2σ, -1σ, median, +1σ, +2σ].
    """
    import pyhf.infer

    model, data = search.to_pyhf_model(signals)
    mu_values = np.linspace(poi_range[0], poi_range[1], n_points)

    cls_obs_list = []
    cls_exp_rows = []  # each row is [−2σ, −1σ, median, +1σ, +2σ]

    for mu in mu_values:
        cls_obs, cls_exp_set = pyhf.infer.hypotest(
            float(mu),
            data,
            model,
            test_stat="qtilde",
            return_expected_set=True,
        )
        cls_obs_list.append(float(cls_obs))
        cls_exp_rows.append([float(v) for v in cls_exp_set])

    cls_obs_arr = np.array(cls_obs_list)
    cls_exp_arr = np.array(cls_exp_rows)  # (n_points, 5)

    return mu_values, cls_obs_arr, [cls_exp_arr[:, i] for i in range(5)]


def plot_cls_scan(
    search: "SearchInput",
    signals: dict[str, float],
    eps2_ref: float | None = None,
    mass: float | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    poi_range: tuple[float, float] = (0.0, 10.0),
    n_points: int = 50,
) -> "matplotlib.axes.Axes":
    """1D CLs scan at a fixed A′ mass hypothesis.

    Scans the signal strength μ (or ε² if ``eps2_ref`` is provided) and
    plots the observed and expected CLs curves with ±1σ / ±2σ bands and
    a horizontal reference line at CLs = 0.05.

    Parameters
    ----------
    search:
        Search configuration (background, observed, systematics).
    signals:
        Expected signal yield per region at the reference ε².
    eps2_ref:
        If provided, the x-axis is displayed in units of ε² = μ × eps2_ref.
    mass:
        A′ mass in MeV, used only for the plot title.
    ax:
        Existing Axes to draw on.  A new figure is created if ``None``.
    poi_range:
        (min, max) range of μ values to scan.
    n_points:
        Number of scan points.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_ax(ax)

    mu_values, cls_obs, cls_exp_band = compute_cls_scan(
        search, signals, poi_range=poi_range, n_points=n_points
    )

    if eps2_ref is not None:
        x = mu_values * eps2_ref
        xlabel = r"$\varepsilon^2$"
    else:
        x = mu_values
        xlabel = r"$\mu$"

    ax.fill_between(
        x, cls_exp_band[0], cls_exp_band[4],
        color="gold", alpha=0.6, label=r"Exp. $\pm 2\sigma$", zorder=1,
    )
    ax.fill_between(
        x, cls_exp_band[1], cls_exp_band[3],
        color="limegreen", alpha=0.8, label=r"Exp. $\pm 1\sigma$", zorder=2,
    )
    ax.plot(x, cls_exp_band[2], color="black", linestyle="--",
            linewidth=1.5, label="Expected", zorder=3)
    ax.plot(x, cls_obs, color="black", linestyle="-",
            linewidth=2.0, label="Observed", zorder=4)
    ax.axhline(0.05, color="red", linestyle=":", linewidth=1.5,
               label=r"CL$_s$ = 0.05", zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"CL$_s$")
    ax.set_ylim(0, 1)

    title = r"CL$_s$ scan"
    if mass is not None:
        title += rf" — $m_{{A'}}$ = {mass:.1f} MeV"
    ax.set_title(title)
    ax.legend()

    return ax


# ---------------------------------------------------------------------------
# Brazil-band helper for (mass, ε) contour plots
# ---------------------------------------------------------------------------


def _mu1_crossings_eps(
    eps2_arr: np.ndarray,
    mu_arr: np.ndarray,
    coup: object,
) -> list[float]:
    """Return ε values where mu_arr crosses 1.0 (log-space interpolation)."""
    crossings: list[float] = []
    for j in range(len(mu_arr) - 1):
        m0, m1 = mu_arr[j], mu_arr[j + 1]
        if np.isnan(m0) or np.isnan(m1):
            continue
        if (m0 - 1.0) * (m1 - 1.0) < 0:
            t = (1.0 - m0) / (m1 - m0)
            log_e = np.log(eps2_arr[j]) + t * (np.log(eps2_arr[j + 1]) - np.log(eps2_arr[j]))
            crossings.append(float(coup(np.exp(log_e))))
    return crossings


def _band_grids(scan_results, value_attr: str, all_eps2: np.ndarray,
                masses: np.ndarray) -> dict[int, np.ndarray]:
    """Build (n_mass, n_eps2) grids of the expected-band test value per σ level.

    ``value_attr`` selects the per-result band array (``mu_upper_exp_band_arr``
    for μ, ``cls_exp_arr`` for CLs); both have shape (n_eps2, 5) with columns
    [−2σ, −1σ, median, +1σ, +2σ].  Returns grids for the four band edges
    (0, 1, 3, 4) used to shade the ±1σ/±2σ Brazil band.
    """
    grids = {bi: np.full((len(masses), len(all_eps2)), np.nan) for bi in (0, 1, 3, 4)}
    for i, sr in enumerate(scan_results):
        idx = np.searchsorted(all_eps2, sr.eps2_arr)
        band = getattr(sr, value_attr)
        for bi in grids:
            grids[bi][i, idx] = band[:, bi]
    return grids


def _fill_exclusion_band(
    ax, masses: np.ndarray, all_eps2: np.ndarray,
    band_grids: dict[int, np.ndarray], level: float, coupling,
    interpolate_mass: bool = False,
) -> None:
    """Shade a ±1σ/±2σ 'Brazil' band on (mass, ε) axes via layered ``contourf``.

    The excluded region at band level σ is ``value ≤ level`` (μ upper limit ≤ 1,
    or CLs ≤ threshold).  Islands are nested (−2σ largest … +2σ smallest), so
    painting −2σ gold, −1σ green, +1σ gold, +2σ axes-facecolor leaves a ring
    band: [−2σ,−1σ]=gold, [−1σ,+1σ]=green, [+1σ,+2σ]=gold, interior cleared.

    Uses the same 2D grid + matplotlib contour engine as the median/observed
    contour lines, so the band extends exactly as far as those lines (closing
    between mass columns) and renders wherever the island exists.
    """
    masses = np.asarray(masses, float)
    if interpolate_mass and len(masses) >= 2:
        keys = list(band_grids.keys())
        masses, interp = _interp_grids_mass(masses, [band_grids[k] for k in keys])
        band_grids = dict(zip(keys, interp))

    M, E = np.meshgrid(masses, coupling(all_eps2))
    facecolor = ax.get_facecolor()

    def _fill(bi: int, color, zorder: float) -> None:
        g = band_grids.get(bi)
        if g is None or not np.isfinite(g).any():
            return
        gT = g.T
        lo = float(np.nanmin(gT)) - 1.0
        if lo >= level:
            return  # nothing excluded at this σ level
        ax.contourf(M, E, gT, levels=[lo, level], colors=[color], zorder=zorder)

    _fill(0, "gold", 0.5)        # −2σ island
    _fill(1, "limegreen", 0.6)   # −1σ island
    _fill(3, "gold", 0.7)        # +1σ island → restores inner ±2σ ring
    _fill(4, facecolor, 0.8)     # +2σ island → clears interior


def _interp_grids_mass(masses: np.ndarray, grids: list[np.ndarray],
                       n_fine: int = 500) -> tuple[np.ndarray, list[np.ndarray]]:
    """Interpolate several (n_mass, n_eps2) grids onto a common fine mass axis."""
    fine = np.linspace(masses.min(), masses.max(), n_fine)
    out = []
    for g in grids:
        gf = np.full((n_fine, g.shape[1]), np.nan)
        for j in range(g.shape[1]):
            col = g[:, j]
            ok = np.isfinite(col)
            if ok.sum() >= 2:
                gf[:, j] = np.interp(fine, masses[ok], col[ok])
        out.append(gf)
    return fine, out


def plot_mu_upper_band(
    mu_scan_results: list["MuUpperScanResult"],
    ax: "matplotlib.axes.Axes | None" = None,
    level: float = 1.0,
    eps_axis: bool = False,
    interpolate_mass: bool = False,
) -> "matplotlib.axes.Axes":
    """Shade the ±1σ / ±2σ expected μ=``level`` Brazil band in (mass, ε) space.

    Built from the same (mass, ε²) grid and matplotlib contour engine as
    :func:`plot_mu_upper_contour`, so the band aligns with — and extends as far
    as — the median expected contour line, and the ±2σ ring renders wherever
    the island exists (no per-mass ``fill_between`` gaps).

    Parameters
    ----------
    mu_scan_results:
        Output of :func:`~hps_limits.limits.scan_mu_upper_from_estimates`.
    ax:
        Existing Axes to draw on.  A new figure is created if ``None``.
    level:
        The μ value at which exclusion is defined (default 1.0).
    eps_axis:
        If ``True``, y-axis is ε = √ε².
    interpolate_mass:
        If ``True``, refine the grid along the mass axis before shading.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_ax(ax)
    _coup = np.sqrt if eps_axis else lambda x: x
    masses = np.array([sr.mass for sr in mu_scan_results])
    all_eps2 = np.unique(np.concatenate([sr.eps2_arr for sr in mu_scan_results]))
    grids = _band_grids(mu_scan_results, "mu_upper_exp_band_arr", all_eps2, masses)
    _fill_exclusion_band(ax, masses, all_eps2, grids, level, _coup, interpolate_mass)
    return ax


# ---------------------------------------------------------------------------
# 1D μ upper limit vs ε² (displaced-vertex direct scan)
# ---------------------------------------------------------------------------


def plot_mu_upper_scan(
    mu_scan: "MuUpperScanResult",
    ax: "matplotlib.axes.Axes | None" = None,
    show_band: bool = True,
    eps_axis: bool = False,
    cl_level: float = 0.95,
) -> "matplotlib.axes.Axes":
    """Plot the μ upper limit as a function of ε² for one mass point.

    Uses the output of :func:`~hps_limits.limits.compute_mu_upper_scan` or
    :func:`~hps_limits.limits.scan_mu_upper_grid`.  The y-axis shows how many
    times the nominal signal cross section would be needed to exclude at 95%
    CL.  The horizontal dashed line at μ = 1 marks the threshold; the region
    below it is excluded.

    Parameters
    ----------
    mu_scan:
        Result for a single mass point.
    ax:
        Existing Axes to draw on.  A new figure is created if ``None``.
    show_band:
        Whether to shade the ±1σ / ±2σ expected bands.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_ax(ax)

    eps2 = np.sqrt(mu_scan.eps2_arr) if eps_axis else mu_scan.eps2_arr
    mu_obs = mu_scan.mu_upper_obs_arr
    mu_exp = mu_scan.mu_upper_exp_arr
    band = mu_scan.mu_upper_exp_band_arr  # (n, 5)

    if show_band:
        ax.fill_between(
            eps2, band[:, 0], band[:, 4],
            color="gold", alpha=0.6, label=r"Exp. $\pm 2\sigma$", zorder=1,
        )
        ax.fill_between(
            eps2, band[:, 1], band[:, 3],
            color="limegreen", alpha=0.8, label=r"Exp. $\pm 1\sigma$", zorder=2,
        )

    cl_pct = int(round(cl_level * 100))
    ax.plot(eps2, mu_exp, color="black", linestyle="--",
            linewidth=1.5, label=f"Expected {cl_pct}% CL", zorder=3)
    ax.plot(eps2, mu_obs, color="black", linestyle="-",
            linewidth=2.0, label=f"Observed {cl_pct}% CL", zorder=4)
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1.5,
               label=r"$\mu = 1$ (nominal)", zorder=5)

    ax.set_xlabel(r"$\varepsilon$" if eps_axis else r"$\varepsilon^2$")
    ax.set_ylabel(rf"$\mu$ upper limit ({cl_pct}% CL)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    title = r"Signal strength upper limit"
    if mu_scan.mass is not None:
        title += rf" — $m_{{A'}}$ = {mu_scan.mass:.0f} MeV"
    ax.set_title(title)
    ax.legend()

    return ax


def _interp_grid_mass(masses: np.ndarray, grid: np.ndarray,
                      n_fine: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a (n_mass, n_eps2) grid to a finer mass axis.

    For each ε² column, linear interpolation is applied along the mass axis
    using only finite values.  Returns the fine mass array and interpolated grid.
    """
    masses_fine = np.linspace(masses.min(), masses.max(), n_fine)
    grid_fine = np.full((n_fine, grid.shape[1]), np.nan)
    for j in range(grid.shape[1]):
        col = grid[:, j]
        valid = np.isfinite(col)
        if valid.sum() >= 2:
            grid_fine[:, j] = np.interp(masses_fine, masses[valid], col[valid])
    return masses_fine, grid_fine


def plot_cls_contour(
    cls_scan_results: list["ClsScanResult"],
    ax: "matplotlib.axes.Axes | None" = None,
    which: str = "obs",
    cl_level: float = 0.95,
    eps_axis: bool = False,
    interpolate_mass: bool = False,
    **contour_kwargs,
) -> "matplotlib.axes.Axes":
    """Draw the CLs exclusion boundary on a (mass, ε) axes using ``ax.contour``.

    Builds the full 2D CLs grid and draws a contour at the exclusion threshold
    ``1 - cl_level``.  Unlike :func:`plot_island_exclusion`, this does not
    require both an upper and lower ε crossing at every mass — the contour is
    traced smoothly across the entire (mass, ε) plane.

    Parameters
    ----------
    cls_scan_results:
        Output of :func:`~hps_limits.limits.scan_cls_from_estimates`.
    ax:
        Existing Axes to draw on.  A new figure is created if ``None``.
    which:
        ``"obs"`` (default) for observed CLs, ``"exp"`` for expected median.
    cl_level:
        Confidence level; contour is drawn at CLs = 1 − ``cl_level``.
    eps_axis:
        If ``True``, the y-axis is ε = √ε².
    interpolate_mass:
        If ``True`` (default), interpolate the CLs grid to a fine mass axis
        before contouring, filling gaps between sparse mass grid points.
    **contour_kwargs:
        Forwarded to ``ax.contour`` (e.g. ``colors``, ``linewidths``,
        ``linestyles``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_ax(ax)
    threshold = 1.0 - cl_level
    _coupling = np.sqrt if eps_axis else lambda x: x

    masses  = np.array([sr.mass for sr in cls_scan_results])
    all_eps2 = np.unique(np.concatenate([sr.eps2_arr for sr in cls_scan_results]))

    grid = np.full((len(masses), len(all_eps2)), np.nan)
    for i, sr in enumerate(cls_scan_results):
        cls_arr = sr.cls_obs_arr if which == "obs" else sr.cls_exp_arr[:, 2]
        idx = np.searchsorted(all_eps2, sr.eps2_arr)
        grid[i, idx] = cls_arr

    if interpolate_mass and len(masses) >= 2:
        masses, grid = _interp_grid_mass(masses, grid)

    mass_c, eps2_c = np.meshgrid(masses, _coupling(all_eps2))
    ax.contour(mass_c, eps2_c, grid.T, levels=[threshold], **contour_kwargs)

    return ax


def plot_cls_band(
    cls_scan_results: list["ClsScanResult"],
    ax: "matplotlib.axes.Axes | None" = None,
    cl_level: float = 0.95,
    eps_axis: bool = False,
    interpolate_mass: bool = False,
) -> "matplotlib.axes.Axes":
    """Shade the ±1σ / ±2σ expected exclusion (Brazil) band on a CLs island plot.

    The excluded region at band level σ is ``CLs ≤ 1 − cl_level``.  Uses the
    same grid + contour engine as :func:`plot_cls_contour`, so the band aligns
    with the median expected contour.  Draw this *before* the contour lines so
    they sit on top.

    Parameters
    ----------
    cls_scan_results:
        Output of :func:`~hps_limits.limits.scan_cls_from_estimates`.
    ax:
        Existing Axes to draw on.  A new figure is created if ``None``.
    cl_level:
        Confidence level; exclusion is CLs ≤ 1 − ``cl_level``.
    eps_axis:
        If ``True``, the y-axis is ε = √ε².
    interpolate_mass:
        If ``True``, refine the grid along the mass axis before shading.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_ax(ax)
    _coupling = np.sqrt if eps_axis else lambda x: x
    threshold = 1.0 - cl_level
    masses = np.array([sr.mass for sr in cls_scan_results])
    all_eps2 = np.unique(np.concatenate([sr.eps2_arr for sr in cls_scan_results]))
    grids = _band_grids(cls_scan_results, "cls_exp_arr", all_eps2, masses)
    _fill_exclusion_band(ax, masses, all_eps2, grids, threshold, _coupling,
                         interpolate_mass)
    return ax


def plot_mu_upper_contour(
    mu_scan_results: list["MuUpperScanResult"],
    ax: "matplotlib.axes.Axes | None" = None,
    which: str = "exp",
    level: float = 1.0,
    eps_axis: bool = False,
    interpolate_mass: bool = False,
    **contour_kwargs,
) -> "matplotlib.axes.Axes":
    """Draw a single μ = ``level`` contour on (mass, ε²) axes.

    Uses the same 2D grid and ``ax.contour`` call as :func:`plot_mu_upper_2d`,
    so the line is identical to the white exclusion boundary drawn there.
    Intended for overlaying per-region contours on a single axes.

    Parameters
    ----------
    mu_scan_results:
        Output of :func:`~hps_limits.limits.scan_mu_upper_from_estimates`.
    ax:
        Existing Axes to draw on.  A new figure is created if ``None``.
    which:
        ``"exp"`` (default) for the expected median limit, ``"obs"`` for
        the observed.
    level:
        The μ value at which to draw the contour (default 1.0).
    eps_axis:
        If ``True``, the y-axis is in ε = √ε² rather than ε².
    **contour_kwargs:
        Forwarded to ``ax.contour`` (e.g. ``colors``, ``linewidths``,
        ``linestyles``, ``label``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_ax(ax)
    _coupling = np.sqrt if eps_axis else lambda x: x

    masses = np.array([sr.mass for sr in mu_scan_results])
    all_eps2 = np.unique(
        np.concatenate([sr.eps2_arr for sr in mu_scan_results])
    )

    grid = np.full((len(masses), len(all_eps2)), np.nan)
    for i, sr in enumerate(mu_scan_results):
        mu_arr = sr.mu_upper_obs_arr if which == "obs" else sr.mu_upper_exp_arr
        idx = np.searchsorted(all_eps2, sr.eps2_arr)
        grid[i, idx] = mu_arr

    if interpolate_mass and len(masses) >= 2:
        masses, grid = _interp_grid_mass(masses, grid)

    mass_c, eps2_c = np.meshgrid(masses, _coupling(all_eps2))
    ax.contour(mass_c, eps2_c, grid.T, levels=[level], **contour_kwargs)

    return ax


def extract_mu_contour(
    mu_scan_results: list["MuUpperScanResult"],
    which: str = "obs",
    level: float = 1.0,
    eps_axis: bool = True,
) -> list[np.ndarray]:
    """Extract the μ=``level`` contour as a list of (N, 2) vertex arrays.

    Uses the same 2D grid as :func:`plot_mu_upper_contour`.  Returns one
    array per disconnected contour segment; each array has shape ``(N, 2)``
    with columns ``[mass_MeV, eps]`` (or ``[mass_MeV, eps²]`` when
    ``eps_axis=False``).

    Parameters
    ----------
    mu_scan_results:
        Output of :func:`~hps_limits.limits.scan_mu_upper_from_estimates`.
    which:
        ``"obs"`` (default) or ``"exp"``.
    level:
        μ value at which to extract the contour (default 1.0).
    eps_axis:
        If ``True`` (default) the second column is ε = √ε²; otherwise ε².

    Returns
    -------
    list of np.ndarray, each shape (N, 2)
        Empty list if no contour exists at the requested level.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _coupling = np.sqrt if eps_axis else lambda x: x

    masses  = np.array([sr.mass for sr in mu_scan_results])
    all_eps2 = np.unique(np.concatenate([sr.eps2_arr for sr in mu_scan_results]))

    grid = np.full((len(masses), len(all_eps2)), np.nan)
    for i, sr in enumerate(mu_scan_results):
        mu_arr = sr.mu_upper_obs_arr if which == "obs" else sr.mu_upper_exp_arr
        idx = np.searchsorted(all_eps2, sr.eps2_arr)
        grid[i, idx] = mu_arr

    mass_c, eps2_c = np.meshgrid(masses, _coupling(all_eps2))

    fig, ax = plt.subplots()
    cs = ax.contour(mass_c, eps2_c, grid.T, levels=[level])
    plt.close(fig)

    return [p.vertices.copy() for p in cs.get_paths()]


def load_exclusion_txt(path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column (mass_MeV, eps) exclusion curve from a text file.

    Lines beginning with ``#`` are treated as comments.  The file may be
    whitespace- or comma-delimited.  Returns ``(masses, eps)`` as 1-D arrays,
    sorted by mass.

    Parameters
    ----------
    path:
        Path to the text file.

    Returns
    -------
    (masses, eps) — both in the same units as the file (MeV and dimensionless ε).
    """
    from pathlib import Path as _Path
    import re as _re

    rows = []
    with open(_Path(path)) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = _re.split(r"[,\s]+", line)
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))

    rows.sort(key=lambda r: r[0])
    masses = np.array([r[0] for r in rows])
    eps    = np.array([r[1] for r in rows])
    return masses, eps


def plot_summary_exclusion(
    mu_scan_results: list["MuUpperScanResult"],
    prior_curves: list[dict] | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    which: str = "obs",
    eps_axis: bool = True,
    new_label: str | None = None,
    new_color: str = "black",
    new_linestyle: str = "-",
    new_linewidth: float = 2.0,
) -> "matplotlib.axes.Axes":
    """Summary exclusion plot: μ=1 contour from a scan + prior limit curves.

    Each prior curve is supplied as a dict with keys:

    ``"masses"`` / ``"eps"``
        Pre-loaded arrays (MeV and ε).  Mutually exclusive with ``"file"``.
    ``"file"``
        Path to a two-column (mass_MeV, eps) text file loaded with
        :func:`load_exclusion_txt`.
    ``"label"`` *(optional)*
        Legend entry.
    ``"color"`` *(optional)*
        Line colour.  Defaults cycle through ``tab10``.
    ``"linestyle"`` *(optional)*
        Defaults to ``"--"``.
    ``"linewidth"`` *(optional)*
        Defaults to ``1.5``.

    Parameters
    ----------
    mu_scan_results:
        Output of :func:`~hps_limits.limits.scan_mu_upper_from_estimates`.
        The μ=1 contour is extracted and drawn on top of the prior curves.
    prior_curves:
        List of prior exclusion dicts (see above).  ``None`` or empty → skip.
    ax:
        Existing Axes to draw on.  A new figure is created if ``None``.
    which:
        ``"obs"`` (default) or ``"exp"`` — selects which μ contour to draw.
    eps_axis:
        If ``True`` (default) the y-axis shows ε = √ε².
    new_label:
        Legend label for the new result's μ=1 contour.
    new_color, new_linestyle, new_linewidth:
        Style for the new result line.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import itertools

    ax = _get_ax(ax)

    _default_colors = itertools.cycle([
        "tab:blue", "tab:orange", "tab:green", "tab:red",
        "tab:purple", "tab:brown", "tab:pink", "tab:gray",
    ])

    legend_handles = []

    # ── Prior curves ──────────────────────────────────────────────────────────
    for curve in (prior_curves or []):
        if "file" in curve:
            masses_p, eps_p = load_exclusion_txt(curve["file"])
        else:
            masses_p = np.asarray(curve["masses"])
            eps_p    = np.asarray(curve["eps"])

        col  = curve.get("color") or next(_default_colors)
        ls   = curve.get("linestyle", "--")
        lw   = curve.get("linewidth", 1.5)
        lbl  = curve.get("label", "")

        ax.plot(masses_p, eps_p, color=col, linestyle=ls, linewidth=lw)
        if lbl:
            legend_handles.append(
                Line2D([0], [0], color=col, linestyle=ls, linewidth=lw, label=lbl)
            )

    # ── New μ=1 contour ───────────────────────────────────────────────────────
    if mu_scan_results:
        plot_mu_upper_contour(
            mu_scan_results, ax=ax, which=which, level=1.0,
            eps_axis=eps_axis,
            colors=[new_color], linewidths=[new_linewidth],
            linestyles=[new_linestyle],
        )
        if new_label:
            legend_handles.append(
                Line2D([0], [0], color=new_color, linestyle=new_linestyle,
                       linewidth=new_linewidth, label=new_label)
            )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel(r"$m_{A'}$ (MeV)")
    ax.set_ylabel(r"$\varepsilon$ (95% CL)" if eps_axis else r"$\varepsilon^2$ (95% CL)")
    ax.set_yscale("log")

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=10)

    return ax


def plot_mu_upper_2d(
    mu_scan_results: list["MuUpperScanResult"],
    ax: "matplotlib.axes.Axes | None" = None,
    which: str = "obs",
    vmin: float | None = None,
    vmax: float | None = None,
    draw_contour: bool = True,
    cmap: str = "jet",
    eps_axis: bool = False,
    cl_level: float = 0.95,
    experiment_label: str | None = None,
    lumi_label: str | None = None,
    y_max: float | None = None,
    n_levels: int | None = None,
) -> "matplotlib.axes.Axes":
    """2D colour map of the μ upper limit in (mass, ε²) space.

    The z-axis (colour) shows how many times the nominal signal cross section
    would be needed to exclude at 95% CL.  Values below 1 are excluded.
    A contour at μ = 1 outlines the island of exclusion.

    Parameters
    ----------
    mu_scan_results:
        Output of :func:`~hps_limits.limits.scan_mu_upper_grid`.
    ax:
        Existing Axes to draw on.  A new figure is created if ``None``.
    which:
        ``"obs"`` (default) for the observed limit, ``"exp"`` for the expected
        median.
    vmin, vmax:
        Colour scale limits (in μ units).  Defaults to the 2nd and 98th
        percentile of finite values so that a single outlier doesn't wash out
        the colour range.
    draw_contour:
        Draw a white contour line at μ = 1.
    cmap:
        Matplotlib colormap name (default ``"viridis_r"`` so low μ = bright).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.colors as mcolors

    ax = _get_ax(ax)

    # Build a regular (mass × eps2) grid.  The ε² axis is taken from the
    # first scan result; all masses should share the same ε² grid.
    masses = np.array([sr.mass for sr in mu_scan_results])

    # Align all scans to a common ε² grid (union, sorted)
    all_eps2 = np.unique(np.concatenate([sr.eps2_arr for sr in mu_scan_results]))

    n_mass = len(masses)
    n_eps2 = len(all_eps2)
    grid = np.full((n_mass, n_eps2), np.nan)

    for i, sr in enumerate(mu_scan_results):
        mu_arr = sr.mu_upper_obs_arr if which == "obs" else sr.mu_upper_exp_arr
        # Map sr.eps2_arr indices onto all_eps2
        idx = np.searchsorted(all_eps2, sr.eps2_arr)
        grid[i, idx] = mu_arr

    # ── ROOT-style rendering: rainbow filled bands + cycling-dash contours ────
    # (matches the collaboration's published limit plots: log colz rainbow with
    # black contour lines cycling through solid/dashed/dash-dot/dotted.)
    #
    # Colour range defaults to the decade bounds enclosing the finite data, so
    # every value is inside [lo, hi] — no colorbar "extend" arrows are needed
    # (flat ends), and the ticks land on clean powers of ten.
    finite = grid[np.isfinite(grid) & (grid > 0)]
    dmin = float(finite.min()) if len(finite) else 1.0
    dmax = float(finite.max()) if len(finite) else 1.0e4
    lo = float(vmin) if vmin is not None else 10.0 ** np.floor(np.log10(dmin))
    hi = float(vmax) if vmax is not None else 10.0 ** np.ceil(np.log10(dmax))
    lo = max(lo, 1e-12)
    if hi <= lo:
        hi = lo * 10.0
    norm = mcolors.LogNorm(vmin=lo, vmax=hi)

    ndec = max(1, int(round(np.log10(hi) - np.log10(lo))))
    nlev = int(n_levels) if n_levels is not None else 5 * ndec + 1
    levels = np.logspace(np.log10(lo), np.log10(hi), nlev)

    _coupling = np.sqrt if eps_axis else lambda x: x
    mass_c, eps2_c = np.meshgrid(masses, _coupling(all_eps2))
    Z = np.clip(grid.T, lo, hi)   # keep all data inside range → no extend arrows

    cf = ax.contourf(mass_c, eps2_c, Z, levels=levels, norm=norm, cmap=cmap)

    cl_pct = int(round(cl_level * 100))
    cbar = ax.get_figure().colorbar(cf, ax=ax)
    cbar.set_label(rf"$\mu$ upper limit ({cl_pct}% CL)")
    # decade ticks: 10^0, 10^1, 10^2, ...
    k0 = int(np.floor(np.log10(lo) + 1e-9))
    k1 = int(np.ceil(np.log10(hi) - 1e-9))
    decade_ticks = [10.0 ** k for k in range(k0, k1 + 1)]
    cbar.set_ticks(decade_ticks)
    cbar.set_ticklabels([rf"$10^{{{k}}}$" for k in range(k0, k1 + 1)])

    if draw_contour:
        # black contour lines at the band edges, cycling ROOT-like dash styles
        _dash = ["-", "--", "-.", ":"]
        lss = [_dash[i % len(_dash)] for i in range(len(levels))]
        ax.contour(mass_c, eps2_c, Z, levels=levels, colors="black",
                   linewidths=0.8, linestyles=lss)

    # HPS-style annotations (top-right, over the dark high-μ region)
    if experiment_label:
        ax.text(0.96, 0.94, experiment_label, transform=ax.transAxes,
                ha="right", va="top", color="white",
                fontstyle="italic", fontweight="bold", fontsize=13)
    if lumi_label:
        ax.text(0.96, 0.87, lumi_label, transform=ax.transAxes,
                ha="right", va="top", color="white",
                fontstyle="italic", fontsize=10)

    ax.set_xlabel(r"$m_{A'}$ (MeV)")
    ax.set_ylabel(r"$\varepsilon$" if eps_axis else r"$\varepsilon^2$")
    ax.set_yscale("log")
    if y_max is not None:
        ax.set_ylim(top=float(y_max))

    if not experiment_label:
        label = "Observed" if which == "obs" else "Expected"
        ax.set_title(rf"{label} {cl_pct}% CL $\mu$ upper limit")

    return ax
