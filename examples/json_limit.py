"""Example: 2D CLs limit from background + signal JSON files.

Loads background estimates (ABCD method) and signal yields from JSON files,
then runs the μ upper-limit scan across all common (mass, ε²) points and
produces the 2D colour-map limit plot.

Optionally also runs the CLs scan for the island-of-exclusion plot.

Usage::

    python3.9 examples/json_limit.py           # full run (all 55 masses)
    python3.9 examples/json_limit.py --cls-only  # CLs scan only (faster per-point)
    python3.9 examples/json_limit.py --mu-only   # μ scan only (produces 2D plot)
"""

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from hps_limits import (
    load_background_json,
    load_signal_json,
    scan_cls_from_estimates,
    scan_mu_upper_from_estimates,
)
from hps_limits.plotting import (
    plot_island_exclusion,
    plot_mu_upper_2d,
    plot_mu_upper_scan,
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--cls-only", action="store_true", help="Run CLs scan only")
parser.add_argument("--mu-only",  action="store_true", help="Run μ scan only")
parser.add_argument("--mu-cap", type=float, default=10000.0,
                    help="Cap on reported μ_upper (default 10000)")
args = parser.parse_args()

run_cls = not args.mu_only
run_mu  = not args.cls_only

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

bg  = load_background_json(DATA_DIR / "background.json")
sig = load_signal_json(DATA_DIR / "signal.json")

print(f"Background: {len(bg.entries)} entries  regions={bg.regions}  {len(bg.masses_mev)} masses")
print(f"Signal:     {len(sig.entries)} entries  regions={sig.regions}")
print(f"            {len(sig.masses_mev)} masses  {sig.masses_mev[0]:.0f}–{sig.masses_mev[-1]:.0f} MeV")
print(f"            ε² = {sig.eps2_values}")

n_grid = len(sig.masses_mev) * len(sig.eps2_values)
print(f"\nGrid size: {len(sig.masses_mev)} masses × {len(sig.eps2_values)} ε² points = {n_grid} points total")

# ---------------------------------------------------------------------------
# CLs scan  →  island-of-exclusion plot
# ---------------------------------------------------------------------------

if run_cls:
    print(f"\n{'='*60}")
    print(f"Running CLs scan ({n_grid} hypotest calls) ...")
    print(f"{'='*60}")
    cls_results = scan_cls_from_estimates(bg, sig, verbose=True)
    print(f"\nCLs scan complete: {len(cls_results)} mass points")

    # Summary table
    print(f"\n{'Mass':>8}  {'Obs ε² upper':>14}  {'Exp ε² upper':>14}")
    print("-" * 42)
    for sr in cls_results:
        obs = sr.limit.eps2_upper_obs
        exp = sr.limit.eps2_upper_exp
        obs_str = f"{obs:.3e}" if obs and not math.isnan(obs) else "---"
        exp_str = f"{exp:.3e}" if exp and not math.isnan(exp) else "---"
        print(f"{sr.mass:>8.0f}  {obs_str:>14}  {exp_str:>14}")

    # Island-of-exclusion plot
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_island_exclusion(cls_results, ax=ax)
    ax.set_title("HPS — Island of Exclusion (95% CL, dummy signal)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "json_exclusion_island.png", dpi=150)
    print(f"\nSaved {OUT_DIR / 'json_exclusion_island.png'}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# μ upper-limit scan  →  2D colour-map plot
# ---------------------------------------------------------------------------

if run_mu:
    print(f"\n{'='*60}")
    print(f"Running μ upper-limit scan ({n_grid} upper_limit calls, cap={args.mu_cap:.0f}) ...")
    print(f"This may take several minutes.")
    print(f"{'='*60}")
    mu_results = scan_mu_upper_from_estimates(
        bg, sig, mu_upper_cap=args.mu_cap, verbose=True
    )
    print(f"\nμ scan complete: {len(mu_results)} mass points")

    # 2D colour-map (observed + expected side by side)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    plot_mu_upper_2d(mu_results, ax=axes2[0], which="obs")
    plot_mu_upper_2d(mu_results, ax=axes2[1], which="exp")
    axes2[0].set_title(r"Observed $\mu$ upper limit (95% CL)")
    axes2[1].set_title(r"Expected $\mu$ upper limit (95% CL)")
    fig2.suptitle("HPS — μ upper limit (dummy signal)", y=1.02)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "json_mu_upper_2d.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'json_mu_upper_2d.png'}")
    plt.close(fig2)

    # Per-mass 1D μ vs ε² plots
    n_masses = len(mu_results)
    ncols = min(5, n_masses)
    nrows = (n_masses + ncols - 1) // ncols
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, mr in enumerate(mu_results):
        row, col = divmod(idx, ncols)
        plot_mu_upper_scan(mr, ax=axes3[row][col])
    for idx in range(n_masses, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes3[row][col].set_visible(False)
    fig3.suptitle(r"HPS — $\mu$ upper limit vs $\varepsilon^2$ (dummy signal)", y=1.01)
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "json_mu_upper_scan.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'json_mu_upper_scan.png'}")
    plt.close(fig3)
