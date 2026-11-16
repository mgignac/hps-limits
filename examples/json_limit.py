"""Example: 2D CLs limit from combined ABCD / z0-binned results JSON files.

Demonstrates two input modes:

1. **Classic** (--mode classic, default): loads background from a combined
   ABCD results JSON and signal from single-bin cut-scan JSONs.

2. **z0-binned** (--mode z0): loads background *and* signal together from
   z0-binned cut-scan JSONs where each z0 bin is a separate pyhf channel,
   enabling a shape fit in the fit variable (min_y0).

Per-region breakdown is run by default alongside the combination.  Use
``--no-per-region`` to skip it.

Usage::

    python3.9 examples/json_limit.py                    # classic mode, full run
    python3.9 examples/json_limit.py --mode z0          # z0-binned mode
    python3.9 examples/json_limit.py --cls-only         # CLs scan only
    python3.9 examples/json_limit.py --mu-only          # μ scan only
    python3.9 examples/json_limit.py --no-per-region    # combined only
    python3.9 examples/json_limit.py --simps            # coupling as ε (not ε²)
    python3.9 examples/json_limit.py --bkg-zero-floor   # systematic: floor zero bkg at 100 events
"""

import argparse
import math
from collections import defaultdict
from itertools import cycle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

import sys

from hps_limits import (
    load_results_json,
    load_cut_scan_jsons,
    load_z0_binned_cut_scan_jsons,
    scan_cls_from_estimates,
    scan_mu_upper_from_estimates,
)
from hps_limits.plotting import (
    plot_island_exclusion,
    plot_mu_upper_2d,
    plot_mu_upper_contour,
    plot_mu_upper_scan,
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["classic", "z0"], default="classic",
                    help="Input mode: 'classic' (separate bkg+sig files) or "
                         "'z0' (z0-binned shape fit)")
parser.add_argument("--cls-only", action="store_true", help="Run CLs scan only")
parser.add_argument("--mu-only",  action="store_true", help="Run μ scan only")
parser.add_argument("--mu-cap", type=float, default=10000.0,
                    help="Cap on reported μ_upper (default 10000)")
parser.add_argument("--no-per-region", action="store_true",
                    help="Skip per-region breakdown; run combined only")
parser.add_argument("--simps", action="store_true",
                    help="SIMPs mode: express coupling as ε instead of ε²")
parser.add_argument("--bkg-zero-floor", action="store_true",
                    help="Systematic study: replace zero projected background with "
                         "100 events (Asimov obs also set to 100).  Off by default. "
                         "Output filenames get a '_bkgfloor100' suffix.")
args = parser.parse_args()

run_cls      = not args.mu_only
run_mu       = not args.cls_only
per_region   = not args.no_per_region
eps_axis     = args.simps

# Output filename suffix so systematic runs never overwrite nominal results.
suffix = "_bkgfloor100" if args.bkg_zero_floor else ""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

if args.mode == "z0":
    z0_files = sorted(DATA_DIR.glob("results_is*_cut_projSig_2.8000.json"))
    if not z0_files:
        raise FileNotFoundError(
            "No results_is*_cut_projSig_2.8000.json files found in data/."
        )
    bg, sig = load_z0_binned_cut_scan_jsons(z0_files)
    mode_label = "z0-binned"
else:
    bg, _sig  = load_results_json(DATA_DIR / "simps_tight_0p3.json", projection="100% projection")
    cut_files = sorted(DATA_DIR.glob("results_minZ0_0.3200_projSig_2.8000_*L1L1*_cut.json"))
    _bg, sig  = load_cut_scan_jsons(cut_files)
    mode_label = "classic"

# ---------------------------------------------------------------------------
# Background zero-floor systematic
#
# When projecting from a small dataset (e.g. 1%) to 100%, regions with zero
# observed events give Nbkg = 0.  A single-event fluctuation can then shift
# the limit dramatically.  This systematic replaces those zeros with a
# conservative floor of 100 events (and sets Asimov observed = 100 to match).
#
# The threshold is 1e-6 rather than 0 because the z0-binned loader clamps
# true-zero backgrounds to 1e-10 to keep pyhf numerically stable; those
# entries are indistinguishable from genuine zeros and must also be floored.
# ---------------------------------------------------------------------------

_BKG_FLOOR           = 100.0
_BKG_FLOOR_THRESHOLD = 1e-6   # entries with nbkg < this are treated as zero

# ---------------------------------------------------------------------------
# Diagnostic printout: loaded background and signal (shown on pre-floor data)
# ---------------------------------------------------------------------------

coupling_label = "ε" if eps_axis else "ε²"
_fmt = lambda v: f"{np.sqrt(v):.3e}" if eps_axis else f"{v:.3e}"

_W = 72  # table width

def _print_data_summary(bg, sig):
    bg_masses = sorted({e.mass_mev for e in bg.entries})
    by_rm     = {(e.region, e.mass_mev): e for e in bg.entries}

    # ---- Background --------------------------------------------------------
    print("\n" + "=" * _W)
    print("BACKGROUND ESTIMATES")
    print("=" * _W)

    n_floor_total = sum(1 for e in bg.entries if e.nbkg < _BKG_FLOOR_THRESHOLD)
    n_small_total = sum(1 for e in bg.entries
                        if _BKG_FLOOR_THRESHOLD <= e.nbkg < 1.0)
    print(f"  {len(bg.entries)} entries  |  {len(bg.regions)} regions  |  "
          f"{len(bg_masses)} mass points")
    print(f"  Effectively zero (nbkg < {_BKG_FLOOR_THRESHOLD:.0e}): "
          f"{n_floor_total} entries"
          + ("  [will be floored]" if args.bkg_zero_floor else
             "  [pass --bkg-zero-floor to floor these]"))
    print(f"  Near-zero (< 1 event):                   {n_small_total} entries")

    for region in bg.regions:
        r_entries = sorted(
            [e for e in bg.entries if e.region == region],
            key=lambda e: e.mass_mev,
        )
        n_floor_r = sum(1 for e in r_entries if e.nbkg < _BKG_FLOOR_THRESHOLD)
        floored_tag = f"  [{n_floor_r} floored]" if (args.bkg_zero_floor and n_floor_r) else \
                      (f"  [{n_floor_r} will floor]" if n_floor_r else "")
        print(f"\n  {region}{floored_tag}")
        print(f"    {'Mass/MeV':>9}  {'Nbkg':>12}  {'±err':>12}  {'Observed':>10}")
        print(f"    {'-'*9}  {'-'*12}  {'-'*12}  {'-'*10}")
        for e in r_entries:
            flag = "  <-- zero" if e.nbkg < _BKG_FLOOR_THRESHOLD else \
                   "  <-- <1"  if e.nbkg < 1.0 else ""
            print(f"    {e.mass_mev:>9.1f}  {e.nbkg:>12.4g}  "
                  f"{e.nbkg_err:>12.4g}  {e.observed:>10.4g}{flag}")

    # ---- Signal ------------------------------------------------------------
    print("\n" + "=" * _W)
    print("SIGNAL ESTIMATES")
    print("=" * _W)
    print(f"  {len(sig.entries)} entries  |  {len(sig.regions)} regions  |  "
          f"{len(sig.masses_mev)} mass points  |  "
          f"{len(sig.eps2_values)} {coupling_label} points")

    col = f"{'Region':<14}  {'Masses':>6}  {coupling_label+' min':>12}  " \
          f"{coupling_label+' max':>12}  {'Nsig min':>10}  {'Nsig max':>10}"
    print(f"\n  {col}")
    print(f"  {'-'*14}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}")
    for r in sig.regions:
        r_ent     = [e for e in sig.entries if e.region == r]
        masses_r  = sorted({e.mass_mev for e in r_ent})
        eps2_vals = [e.eps2 for e in r_ent]
        nsig_vals = [e.nsig for e in r_ent]
        e2_lo = np.sqrt(min(eps2_vals)) if eps_axis else min(eps2_vals)
        e2_hi = np.sqrt(max(eps2_vals)) if eps_axis else max(eps2_vals)
        print(f"  {r:<14}  {len(masses_r):>6}  {e2_lo:>12.3e}  {e2_hi:>12.3e}  "
              f"{min(nsig_vals):>10.4f}  {max(nsig_vals):>10.4f}")

    print("=" * _W)

print(f"Mode: {mode_label}")
_print_data_summary(bg, sig)

# Apply floor after diagnostic so the table shows raw (pre-floor) values.
if args.bkg_zero_floor:
    n_floored = 0
    for e in bg.entries:
        if e.nbkg < _BKG_FLOOR_THRESHOLD:
            e.nbkg     = _BKG_FLOOR
            e.nbkg_err = math.sqrt(_BKG_FLOOR)
            e.observed = _BKG_FLOOR
            n_floored += 1
    print(f"\n[bkg-zero-floor] Applied: {n_floored} entries set to "
          f"{_BKG_FLOOR:.0f} events (nbkg_err = {math.sqrt(_BKG_FLOOR):.1f})")

n_grid = len(sig.masses_mev) * len(sig.eps2_values)
print(f"\nGrid size: {len(sig.masses_mev)} masses × "
      f"{len(sig.eps2_values)} {coupling_label} points = {n_grid} total")

eps2_range = (min(sig.eps2_values), max(sig.eps2_values))

# ---------------------------------------------------------------------------
# Determine per-region categories
#
# In classic mode channels are named by hit category directly (L1L1, L2L2 …).
# In z0-binned mode channels are named {hit_category}_{bin} (L1L1_A, L1L1_B …);
# we group them back to hit categories so each "region" run uses all z0 bins
# for that category together.
# ---------------------------------------------------------------------------

_common = set(bg.regions) & set(sig.regions)

if args.mode == "z0":
    # Group L1L1_A / L1L1_B / L1L1_C → L1L1, etc.
    _groups: dict[str, list[str]] = defaultdict(list)
    for r in bg.regions:
        if r not in _common:
            continue
        parts = r.rsplit("_", 1)
        cat = parts[0] if len(parts) == 2 and len(parts[1]) <= 2 else r
        _groups[cat].append(r)
    # Preserve original ordering of first appearance
    _seen_cats: dict[str, None] = {}
    for r in bg.regions:
        parts = r.rsplit("_", 1)
        cat = parts[0] if len(parts) == 2 and len(parts[1]) <= 2 else r
        if cat in _groups:
            _seen_cats[cat] = None
    categories: dict[str, list[str]] = {cat: _groups[cat] for cat in _seen_cats}
else:
    categories = {r: [r] for r in bg.regions if r in _common}

print(f"\nRegion categories: {list(categories)}")

# ---------------------------------------------------------------------------
# Color palette (consistent across CLs and μ plots)
# ---------------------------------------------------------------------------

_PALETTE = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]
_color_cycle = cycle(_PALETTE)
REGION_COLOR = {cat: next(_color_cycle) for cat in categories}

# ===========================================================================
# CLs scan
# ===========================================================================

if run_cls:
    print(f"\n{'='*60}")
    print(f"Running CLs scan ...")
    print(f"{'='*60}")

    # --- Combined -----------------------------------------------------------
    print("\n  [Combined]")
    cls_results = scan_cls_from_estimates(bg, sig, verbose=True)
    print(f"  → {len(cls_results)} mass points")

    col_hdr = f"Obs {coupling_label} upper"
    exp_hdr = f"Exp {coupling_label} upper"
    print(f"\n{'Mass':>8}  {col_hdr:>16}  {exp_hdr:>16}")
    print("-" * 46)
    for sr in cls_results:
        obs = sr.limit.eps2_upper_obs
        exp = sr.limit.eps2_upper_exp
        obs_str = _fmt(obs) if obs and not math.isnan(obs) else "---"
        exp_str = _fmt(exp) if exp and not math.isnan(exp) else "---"
        print(f"{sr.mass:>8.0f}  {obs_str:>16}  {exp_str:>16}")

    # Combined island-of-exclusion plot
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_island_exclusion(cls_results, ax=ax, ylim=eps2_range, eps_axis=eps_axis)
    ax.set_title("HPS — Island of Exclusion (95% CL, Combined)")
    fig.tight_layout()
    out = OUT_DIR / f"json_exclusion_island_combined{suffix}.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    plt.close(fig)

    # --- Per-region ---------------------------------------------------------
    if per_region:
        per_region_cls: dict[str, list] = {}

        for cat, chans in categories.items():
            print(f"\n  [{cat}]  channels={chans}")
            per_region_cls[cat] = scan_cls_from_estimates(
                bg, sig, select_regions=chans, verbose=False
            )
            print(f"  → {len(per_region_cls[cat])} mass points")

            # Individual island plot
            fig_r, ax_r = plt.subplots(figsize=(10, 5))
            plot_island_exclusion(
                per_region_cls[cat], ax=ax_r, ylim=eps2_range, eps_axis=eps_axis,
            )
            ax_r.set_title(f"HPS — Island of Exclusion (95% CL, {cat})")
            fig_r.tight_layout()
            out_r = OUT_DIR / f"json_exclusion_island_{cat}{suffix}.png"
            fig_r.savefig(out_r, dpi=150)
            print(f"  Saved {out_r}")
            plt.close(fig_r)

        # Overlay comparison: all regions (expected only) + combined (obs+exp+band)
        fig_cmp, ax_cmp = plt.subplots(figsize=(10, 5))

        for cat, results_r in per_region_cls.items():
            col = REGION_COLOR[cat]
            plot_island_exclusion(
                results_r, ax=ax_cmp, show_band=False, ylim=eps2_range,
                eps_axis=eps_axis,
                obs_style=dict(color=col, linestyle="-",  linewidth=1.2,
                               alpha=0.85, label=f"{cat}"),
                exp_style=dict(color=col, linestyle="--", linewidth=1.0,
                               alpha=0.85),
            )

        plot_island_exclusion(
            cls_results, ax=ax_cmp, show_band=True, ylim=eps2_range,
            eps_axis=eps_axis,
            obs_style=dict(color="black", linestyle="-",  linewidth=2.2,
                           label="Combined (obs.)"),
            exp_style=dict(color="black", linestyle="--", linewidth=1.6,
                           label="Combined (exp.)"),
        )

        ax_cmp.set_title("HPS — Region Comparison: Island of Exclusion (95% CL)")
        ax_cmp.legend(fontsize=8, ncol=2)
        fig_cmp.tight_layout()
        out_cmp = OUT_DIR / f"json_exclusion_island_comparison{suffix}.png"
        fig_cmp.savefig(out_cmp, dpi=150)
        print(f"\nSaved {out_cmp}")
        plt.close(fig_cmp)


# ===========================================================================
# μ upper-limit scan
# ===========================================================================

if run_mu:
    print(f"\n{'='*60}")
    print(f"Running μ upper-limit scan (cap={args.mu_cap:.0f}) ...")
    print(f"This may take several minutes.")
    print(f"{'='*60}")

    model_label = "HPS SIMPs" if eps_axis else "HPS"

    # --- Combined -----------------------------------------------------------
    print("\n  [Combined]")
    mu_results = scan_mu_upper_from_estimates(
        bg, sig, mu_upper_cap=args.mu_cap, verbose=True
    )
    print(f"  → {len(mu_results)} mass points")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    plot_mu_upper_2d(mu_results, ax=axes2[0], which="obs", eps_axis=eps_axis)
    plot_mu_upper_2d(mu_results, ax=axes2[1], which="exp", eps_axis=eps_axis)
    axes2[0].set_title(r"Observed $\mu$ upper limit (95% CL)")
    axes2[1].set_title(r"Expected $\mu$ upper limit (95% CL)")
    fig2.suptitle(f"{model_label} — μ upper limit (Combined)", y=1.02)
    fig2.tight_layout()
    out2 = OUT_DIR / f"json_mu_upper_2d_combined{suffix}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig2)

    # --- Per-region ---------------------------------------------------------
    if per_region:
        per_region_mu: dict[str, list] = {}

        for cat, chans in categories.items():
            print(f"\n  [{cat}]  channels={chans}")
            per_region_mu[cat] = scan_mu_upper_from_estimates(
                bg, sig, select_regions=chans,
                mu_upper_cap=args.mu_cap, verbose=False,
            )
            print(f"  → {len(per_region_mu[cat])} mass points")

            # Individual 2D colourmap
            fig_r2, axes_r2 = plt.subplots(1, 2, figsize=(14, 5))
            plot_mu_upper_2d(per_region_mu[cat], ax=axes_r2[0], which="obs",
                             eps_axis=eps_axis)
            plot_mu_upper_2d(per_region_mu[cat], ax=axes_r2[1], which="exp",
                             eps_axis=eps_axis)
            axes_r2[0].set_title(r"Observed $\mu$ upper limit (95% CL)")
            axes_r2[1].set_title(r"Expected $\mu$ upper limit (95% CL)")
            fig_r2.suptitle(f"{model_label} — μ upper limit ({cat})", y=1.02)
            fig_r2.tight_layout()
            out_r2 = OUT_DIR / f"json_mu_upper_2d_{cat}{suffix}.png"
            fig_r2.savefig(out_r2, dpi=150, bbox_inches="tight")
            print(f"  Saved {out_r2}")
            plt.close(fig_r2)

        # Overlay comparison: μ=1 exclusion contours per region + combined
        # Uses the same ax.contour on the full 2D grid as plot_mu_upper_2d,
        # so lines are identical to the white boundaries in the colourmap plots.
        fig_cmp2, ax_cmp2 = plt.subplots(figsize=(10, 5))

        legend_handles = []
        for cat, results_r in per_region_mu.items():
            col = REGION_COLOR[cat]
            plot_mu_upper_contour(
                results_r, ax=ax_cmp2, which="exp", eps_axis=eps_axis,
                colors=[col], linewidths=[1.4], linestyles=["--"],
            )
            legend_handles.append(
                Line2D([0], [0], color=col, linestyle="--", linewidth=1.4, label=cat)
            )

        plot_mu_upper_contour(
            mu_results, ax=ax_cmp2, which="exp", eps_axis=eps_axis,
            colors=["black"], linewidths=[2.0], linestyles=["--"],
        )
        legend_handles.append(
            Line2D([0], [0], color="black", linestyle="--", linewidth=2.0,
                   label="Combined (exp.)")
        )

        plot_mu_upper_contour(
            mu_results, ax=ax_cmp2, which="obs", eps_axis=eps_axis,
            colors=["black"], linewidths=[2.0], linestyles=["-"],
        )
        legend_handles.append(
            Line2D([0], [0], color="black", linestyle="-", linewidth=2.0,
                   label="Combined (obs.)")
        )

        ax_cmp2.set_xlabel(r"$m_{A'}$ (MeV)")
        ylabel = (r"$\varepsilon$ (95% CL, $\mu=1$ contour)"
                  if eps_axis else r"$\varepsilon^2$ (95% CL, $\mu=1$ contour)")
        ax_cmp2.set_ylabel(ylabel)
        ax_cmp2.set_yscale("log")
        ax_cmp2.set_title(f"{model_label} — Region Comparison: μ=1 Exclusion Contour")
        ax_cmp2.legend(handles=legend_handles, fontsize=11, ncol=2, loc="upper right")
        fig_cmp2.tight_layout()
        out_cmp2 = OUT_DIR / f"json_mu_upper_comparison{suffix}.png"
        fig_cmp2.savefig(out_cmp2, dpi=150)
        print(f"\nSaved {out_cmp2}")
        plt.close(fig_cmp2)

        # Per-mass 1D μ vs coupling: all regions + combined on each subplot
        n_masses = len(mu_results)
        ncols = min(5, n_masses)
        nrows = (n_masses + ncols - 1) // ncols
        fig3, axes3 = plt.subplots(nrows, ncols,
                                   figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for idx, mr_comb in enumerate(mu_results):
            row, col_idx = divmod(idx, ncols)
            ax3 = axes3[row][col_idx]
            mass = mr_comb.mass
            _coup2 = np.sqrt if eps_axis else lambda x: x

            # Individual regions
            for cat, results_r in per_region_mu.items():
                mr_r = next((m for m in results_r if m.mass == mass), None)
                if mr_r is None:
                    continue
                col = REGION_COLOR[cat]
                ax3.plot(_coup2(mr_r.eps2_arr), mr_r.mu_upper_exp_arr,
                         color=col, linestyle="--", linewidth=1.2,
                         alpha=0.85, label=cat)

            # Combined (obs + exp)
            ax3.plot(_coup2(mr_comb.eps2_arr), mr_comb.mu_upper_exp_arr,
                     color="black", linestyle="--", linewidth=1.5,
                     label="Combined (exp.)")
            ax3.plot(_coup2(mr_comb.eps2_arr), mr_comb.mu_upper_obs_arr,
                     color="black", linestyle="-", linewidth=2.0,
                     label="Combined (obs.)")
            ax3.axhline(1.0, color="red", linestyle=":", linewidth=1.2)

            ax3.set_xscale("log")
            ax3.set_yscale("log")
            ax3.set_xlabel(r"$\varepsilon$" if eps_axis else r"$\varepsilon^2$")
            ax3.set_ylabel(r"$\mu$ upper limit")
            ax3.set_title(rf"$m_{{A'}}$ = {mass:.0f} MeV")
            ax3.legend(fontsize=6, ncol=2)

        for idx in range(n_masses, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes3[r][c].set_visible(False)

        coup_tex = r"$\varepsilon$" if eps_axis else r"$\varepsilon^2$"
        fig3.suptitle(rf"{model_label} — $\mu$ upper limit vs {coup_tex} (per region)", y=1.01)
        fig3.tight_layout()
        out3 = OUT_DIR / f"json_mu_upper_scan_per_region{suffix}.png"
        fig3.savefig(out3, dpi=150, bbox_inches="tight")
        print(f"Saved {out3}")
        plt.close(fig3)

    else:
        # No per-region: just the combined 1D scan plots
        n_masses = len(mu_results)
        ncols = min(5, n_masses)
        nrows = (n_masses + ncols - 1) // ncols
        fig3, axes3 = plt.subplots(nrows, ncols,
                                   figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for idx, mr in enumerate(mu_results):
            row, col_idx = divmod(idx, ncols)
            plot_mu_upper_scan(mr, ax=axes3[row][col_idx], eps_axis=eps_axis)
        for idx in range(n_masses, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes3[r][c].set_visible(False)
        coup_tex = r"$\varepsilon$" if eps_axis else r"$\varepsilon^2$"
        fig3.suptitle(rf"{model_label} — $\mu$ upper limit vs {coup_tex}", y=1.01)
        fig3.tight_layout()
        out3 = OUT_DIR / f"json_mu_upper_scan{suffix}.png"
        fig3.savefig(out3, dpi=150, bbox_inches="tight")
        print(f"Saved {out3}")
        plt.close(fig3)
