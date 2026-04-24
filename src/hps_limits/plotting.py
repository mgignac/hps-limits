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

    if not masses:
        ax.set_xlabel(r"$m_{A'}$ (MeV)")
        ax.set_ylabel(r"$\varepsilon^2$ (95% CL island)")
        ax.set_yscale("log")
        return ax

    masses_arr = np.array(masses)

    def _masked(lst):
        """Convert a list (possibly containing nan) to a masked array."""
        a = np.array(lst, dtype=float)
        return np.ma.masked_invalid(a)

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
                      label="Expected 95% CL", zorder=3)
    if exp_style:
        _exp_style.update(exp_style)

    # Draw both boundaries of the expected island
    ax.plot(masses_arr, exp_lo_a, **_exp_style)
    _exp_upper_style = dict(_exp_style)
    _exp_upper_style.pop("label", None)
    ax.plot(masses_arr, exp_hi_a, **_exp_upper_style)

    _obs_style = dict(color="black", linestyle="-", linewidth=2.0,
                      label="Observed 95% CL", zorder=4)
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
    ax.set_ylabel(r"$\varepsilon^2$ (95% CL island)")
    ax.set_yscale("log")
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
# 1D μ upper limit vs ε² (displaced-vertex direct scan)
# ---------------------------------------------------------------------------


def plot_mu_upper_scan(
    mu_scan: "MuUpperScanResult",
    ax: "matplotlib.axes.Axes | None" = None,
    show_band: bool = True,
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

    eps2 = mu_scan.eps2_arr
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

    ax.plot(eps2, mu_exp, color="black", linestyle="--",
            linewidth=1.5, label="Expected 95% CL", zorder=3)
    ax.plot(eps2, mu_obs, color="black", linestyle="-",
            linewidth=2.0, label="Observed 95% CL", zorder=4)
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1.5,
               label=r"$\mu = 1$ (nominal)", zorder=5)

    ax.set_xlabel(r"$\varepsilon^2$")
    ax.set_ylabel(r"$\mu$ upper limit (95% CL)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    title = r"Signal strength upper limit"
    if mu_scan.mass is not None:
        title += rf" — $m_{{A'}}$ = {mu_scan.mass:.0f} MeV"
    ax.set_title(title)
    ax.legend()

    return ax


def plot_mu_upper_2d(
    mu_scan_results: list["MuUpperScanResult"],
    ax: "matplotlib.axes.Axes | None" = None,
    which: str = "obs",
    vmin: float | None = None,
    vmax: float | None = None,
    draw_contour: bool = True,
    cmap: str = "viridis_r",
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

    # Colour limits
    finite = grid[np.isfinite(grid)]
    if vmin is None:
        vmin = float(np.percentile(finite, 2)) if len(finite) else 0.1
    if vmax is None:
        vmax = float(np.percentile(finite, 98)) if len(finite) else 100.0

    norm = mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)

    # pcolormesh needs 2D coordinate arrays on cell edges.
    # Build edge arrays by extending the centre arrays by half a bin.
    def _edges(centres):
        c = np.asarray(centres, dtype=float)
        log_c = np.log(c)
        log_edges = np.concatenate([
            [log_c[0] - 0.5 * (log_c[1] - log_c[0])],
            0.5 * (log_c[:-1] + log_c[1:]),
            [log_c[-1] + 0.5 * (log_c[-1] - log_c[-2])],
        ])
        return np.exp(log_edges)

    mass_edges = np.concatenate([
        [masses[0] - 0.5 * (masses[1] - masses[0])],
        0.5 * (masses[:-1] + masses[1:]),
        [masses[-1] + 0.5 * (masses[-1] - masses[-2])],
    ])
    eps2_edges = _edges(all_eps2)

    # grid shape: (n_mass, n_eps2) → pcolormesh wants (rows=eps2, cols=mass)
    pcm = ax.pcolormesh(
        mass_edges,
        eps2_edges,
        grid.T,
        norm=norm,
        cmap=cmap,
        shading="flat",
    )

    cbar = ax.get_figure().colorbar(pcm, ax=ax)
    cbar.set_label(r"$\mu$ upper limit (95% CL)")

    if draw_contour:
        # contour needs centre coordinates
        mass_c, eps2_c = np.meshgrid(masses, all_eps2)
        ax.contour(
            mass_c, eps2_c, grid.T,
            levels=[1.0],
            colors="white",
            linewidths=2.0,
            linestyles="-",
        )

    ax.set_xlabel(r"$m_{A'}$ (MeV)")
    ax.set_ylabel(r"$\varepsilon^2$")
    ax.set_yscale("log")

    label = "Observed" if which == "obs" else "Expected"
    ax.set_title(rf"{label} 95% CL $\mu$ upper limit")

    return ax
