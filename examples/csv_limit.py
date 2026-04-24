"""Example: Island-of-exclusion and 1D μ upper limit plots from CSV data.

Loads signal and background predictions from the CSV file, runs the full
CLs(ε²) scan at each mass, and produces:
  - An island-of-exclusion plot (2D: mass vs ε²)
  - Per-mass 1D μ upper limit plots (μ_upper vs ε²)

Usage::

    python3.9 examples/csv_limit.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from hps_limits import load_search_csv, scan_cls_grid, scan_mu_upper_grid
from hps_limits.plotting import plot_island_exclusion, plot_mu_upper_scan, plot_mu_upper_2d

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUT_DIR = Path(__file__).parent.parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

csv_path = Path(__file__).parent.parent / "data" / "tight_cuts_displaced_acceptance_l1l2.csv"
points = load_search_csv(csv_path)

masses = sorted({p["mass_mev"] for p in points})
print(f"Loaded {len(points)} (mass, ε²) points across {len(masses)} masses: {masses} MeV")

# ---------------------------------------------------------------------------
# Scale signal yields for testing (remove once real normalisation is applied)
# ---------------------------------------------------------------------------

SIGNAL_SCALE = (180.0 / 1.15) 
for p in points:
    p["nsig"] *= SIGNAL_SCALE
    p["nbkg"] *= 0.001 * SIGNAL_SCALE

# ---------------------------------------------------------------------------
# Run full CLs scan grid
# ---------------------------------------------------------------------------

print("Running CLs scan grid (this may take a minute)...")
scan_results = scan_cls_grid(points, verbose=True)

# ---------------------------------------------------------------------------
# Print summary table (island boundaries)
# ---------------------------------------------------------------------------

print(f"\n{'Mass (MeV)':>12}  {'Obs ε² lower':>14}  {'Obs ε² upper':>14}  {'Exp ε² lower':>14}  {'Exp ε² upper':>14}")
print("-" * 80)
from hps_limits.limits import _find_eps2_crossings
import numpy as np

threshold = 0.05
for sr in scan_results:
    obs_cross = _find_eps2_crossings(sr.eps2_arr, sr.cls_obs_arr, threshold)
    exp_cross = _find_eps2_crossings(sr.eps2_arr, sr.cls_exp_arr[:, 2], threshold)
    obs_lo = f"{obs_cross[0]:.3e}" if len(obs_cross) >= 2 else "---"
    obs_hi = f"{obs_cross[-1]:.3e}" if len(obs_cross) >= 2 else "---"
    exp_lo = f"{exp_cross[0]:.3e}" if len(exp_cross) >= 2 else "---"
    exp_hi = f"{exp_cross[-1]:.3e}" if len(exp_cross) >= 2 else "---"
    print(f"{sr.mass:>12.0f}  {obs_lo:>14}  {obs_hi:>14}  {exp_lo:>14}  {exp_hi:>14}")

# ---------------------------------------------------------------------------
# Island-of-exclusion plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
plot_island_exclusion(scan_results, ax=ax)
ax.set_title(r"HPS Displaced Search — Island of Exclusion (95% CL)"
             f" (Nsig ×{SIGNAL_SCALE:.0f})")
ax.set_xlim(masses[0], masses[-1])
fig.tight_layout()
fig.savefig(OUT_DIR / "exclusion_island.png", dpi=150)
print(f"\nSaved {OUT_DIR / 'exclusion_island.png'}")

# ---------------------------------------------------------------------------
# μ upper limit scan (per mass)
# ---------------------------------------------------------------------------

print("\nRunning μ upper limit scan (this may take a few minutes)...")
mu_results = scan_mu_upper_grid(points, verbose=True)

# One plot per mass
n_masses = len(mu_results)
ncols = min(4, n_masses)
nrows = (n_masses + ncols - 1) // ncols
fig2, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

for idx, mu_res in enumerate(mu_results):
    row, col = divmod(idx, ncols)
    plot_mu_upper_scan(mu_res, ax=axes[row][col])

# Hide unused subplots
for idx in range(n_masses, nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row][col].set_visible(False)

fig2.suptitle(
    r"HPS Displaced Search — $\mu$ upper limit vs $\varepsilon^2$"
    f" (Nsig ×{SIGNAL_SCALE:.0f})",
    y=1.01,
)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "mu_upper_scan.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR / 'mu_upper_scan.png'}")

# ---------------------------------------------------------------------------
# 2D μ upper limit colour map
# ---------------------------------------------------------------------------

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
plot_mu_upper_2d(mu_results, ax=axes3[0], which="obs")
plot_mu_upper_2d(mu_results, ax=axes3[1], which="exp")
fig3.suptitle(
    r"HPS Displaced Search — 95% CL $\mu$ upper limit"
    f" (Nsig ×{SIGNAL_SCALE:.0f})",
)
fig3.tight_layout()
fig3.savefig(OUT_DIR / "mu_upper_2d.png", dpi=150)
print(f"Saved {OUT_DIR / 'mu_upper_2d.png'}")
