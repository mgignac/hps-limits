"""Example: 2D exclusion plot (ε² vs mass) for a toy HPS A′ search.

Generates toy inputs across 50–250 MeV and produces:
  - exclusion.png  — 2D ε² vs mass plot with ±1σ / ±2σ expected bands

Usage::

    python3.9 examples/example_limit.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hps_limits import scan_mass_limits
from hps_limits.plotting import plot_exclusion

# ---------------------------------------------------------------------------
# Toy inputs across 50–250 MeV
#
# Signal yield falls with mass (falling production cross section × acceptance).
# Background and its uncertainty come from a sideband estimate in each
# mass window; both fall at higher mass as the window narrows.
# ---------------------------------------------------------------------------

masses = np.arange(50, 251, 10, dtype=float)  # 50, 60, ..., 250 MeV

m0 = masses[0]  # reference mass for scaling

# Signal yield at reference coupling ε²_ref (falls roughly as 1/m²)
signals = (20.0 * (m0 / masses) ** 2).tolist()

# Background estimate in each mass window (falls roughly as 1/m)
backgrounds = (100.0 * (m0 / masses)).tolist()

# 10% background uncertainty from sideband fit
bkg_uncertainties = (0.10 * np.array(backgrounds)).tolist()

# Reference ε² at which the MC signal yields above were computed
eps2_ref = 1e-6

# ---------------------------------------------------------------------------
# Compute expected-only limits (observed = background)
# ---------------------------------------------------------------------------

results = scan_mass_limits(
    masses=masses.tolist(),
    signals=signals,
    backgrounds=backgrounds,
    background_uncertainties=bkg_uncertainties,
)

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

print(f"{'Mass (MeV)':>12}  {'Obs ε²_95':>12}  {'Exp ε²_95':>12}  {'[-2σ, +2σ]'}")
print("-" * 60)
for r in results:
    b = r.mu_upper_exp_band
    obs_eps2 = r.mu_upper_obs * eps2_ref
    exp_eps2 = r.mu_upper_exp * eps2_ref
    lo_eps2  = b[0] * eps2_ref
    hi_eps2  = b[4] * eps2_ref
    print(f"{r.mass:>12.0f}  {obs_eps2:>12.2e}  {exp_eps2:>12.2e}  [{lo_eps2:.2e}, {hi_eps2:.2e}]")

# ---------------------------------------------------------------------------
# 2D exclusion plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

plot_exclusion(results, eps2_ref=eps2_ref, ax=ax)

ax.set_xlim(masses[0], masses[-1])
ax.set_title("HPS Toy A′ Search — 95% CL Upper Limits")
ax.set_xlabel(r"$m_{A'}$ (MeV)")

fig.tight_layout()
fig.savefig("exclusion.png", dpi=150)
print("\nSaved exclusion.png")
