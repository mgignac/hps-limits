# hps-limits

CLs upper-limit setting for HPS (Heavy Photon Search) A′ / SIMP search results,
built on [`pyhf`](https://pyhf.readthedocs.io/) 0.7.6.

The package builds a multi-channel `pyhf` model from data-driven background
estimates and signal templates, then computes 90% (or any) CL upper limits on a
signal-strength parameter μ over a grid of A′ mass and coupling (ε or ε²). It
produces exclusion-contour and μ-upper-limit plots, including ±1σ/±2σ expected
(Brazil) bands.

---

## Installation

The pinned dependencies live under **Python 3.9** (the system `python3` may be a
newer version that lacks them):

```bash
python3.9 -m pip install -e ".[dev]"
```

This installs `pyhf[contrib]`, `numpy`, `matplotlib`, `pyyaml`, and `pytest`.

Run the test suite with:

```bash
python3.9 -m pytest tests/ -v
```

---

## Quick start

Everything is driven by a YAML config through `scripts/run_limits.py`:

```bash
python3.9 scripts/run_limits.py --config examples/config_z0.yaml
```

Plots and contour text files are written to the config's `output_dir`.

### Command-line overrides

| Flag | Effect |
|------|--------|
| `--cls-only` | Run only the CLs island scan (skip μ scan) |
| `--mu-only` | Run only the μ upper-limit scan (skip CLs) |
| `--no-per-region` | Skip the per-category breakdown |
| `--bkg-zero-floor` | Floor near-zero backgrounds at 100 events |
| `--mu-cap N` | Cap on the reported μ upper limit |
| `--max-mass-points N` | Use only the first N masses (quick tests) |

---

## Analysis modes

Set by `mode` (and, in `classic`, by `signal_format`):

- **`mode: z0`** — z0-binned shape fit. Each hit category is split into bins of a
  displaced-vertex variable; one `pyhf` channel per `(category, z0-bin)`. Signal
  and background are loaded from per-bin ABCD results JSONs. *(SIMP analysis.)*
- **`mode: classic`, `signal_format: cut_scan`** — single channel per hit category
  from a combined results JSON plus a separate background JSON.
- **`mode: classic`, `signal_format: ap_displaced`** — A′ displaced-vertex signal
  yields from mass-keyed JSON files (`ap{mass}MeV → {eps2, yields}`), one per hit
  category. Typically run background-free (`zero_background: true`).

---

## Configuration reference

```yaml
label: "HPS SIMPs"          # title / output-filename prefix
mode: z0                    # z0 | classic
output_dir: ../plots/run1   # resolved relative to the config file

# ── Analysis options ──────────────────────────────────────────────
simps: true                 # true → ε axis (SIMP); false → ε² axis
per_region: true            # also run each hit category individually
run_cls: true               # CLs island-of-exclusion scan
run_mu: true                # μ upper-limit scan
mu_cap: 10000.0             # cap on reported μ (and early stopping)
cl_level: 0.90              # confidence level (0.90 = 90% CL)

# ── Inputs: z0 mode ───────────────────────────────────────────────
z0_files:                   # signal (one file per hit category)
  - ../data/.../results_isL1L1_cut_projSig_2.8000.json
background_jsons:           # per-z0-bin background files (globs OK)
  - ../data/.../results_*_bin*.json
background_projection: null # null = nominal bkg_est + real n_a_obs (unblinded)
                            # or a projection key (e.g. "100% projection") for Asimov
background_z0_labels: [lo, hi]   # bin1→lo, bin2→hi (default)

# ── Inputs: classic / ap_displaced ────────────────────────────────
signal_format: ap_displaced
signal_jsons:               # multi-file ap_displaced
  - {file: ../data/.../ap_displaced_yield__l1l1_...json, region: L1L1}
# background_json: ../data/.../background.json   # or omit with zero_background

# ── Post-load transformations ─────────────────────────────────────
zero_background: true       # background-free search (synthetic zero bkg)
signal_scale: 1.0           # multiply signal yields (e.g. luminosity projection)
background_scale: 1.0       # multiply background + observed yields (flat projection)
background_observed_rate:   # project background from observed counts (see below)
  lumi_ratio: 8.983393389558545

# ── Signal systematics (normsys NPs, correlated across channels) ──
signal_systematics:
  - {name: MC_xsec, value: 0.035}   # value = relative 1σ

# ── Background systematics (ABCD method) ──────────────────────────
background_systematics:
  file: ../data/.../background_systematics.json
  treatment: expected_band  # expected_band | profile
  mode: components          # components | total | total_uncorrelated (profile only)

# ── Plots ─────────────────────────────────────────────────────────
interpolate_contour_mass: false
summary_plot: true
prior_exclusions: []        # list of {label, file, color, linestyle}
bkg_zero_floor: false
```

### Luminosity projection

For a flat projection, multiply yields by the integrated-luminosity ratio:

```yaml
signal_scale:     8.983393389558545   # = L(full)/L(measured), e.g. 161.799/18.0109
background_scale: 8.983393389558545
```

For a **rate-based** projection that uses the measured observed counts instead of
the ABCD estimate (more robust against low-stat/zero bins), use
`background_observed_rate` (mutually exclusive with `background_scale`):

```yaml
signal_scale: 8.983393389558545
background_observed_rate:
  lumi_ratio: 8.983393389558545
```

Per channel/mass: `bkg = obs = n_a_obs × lumi_ratio` (Asimov projection — there is
no real full-lumi data, so the observed limit equals the median expected). Bins
with zero observed counts fall back to `bkg_est × lumi_ratio`.

### Background-systematics treatment

- **`expected_band`** — the ABCD systematic is applied only to the *combined
  expected* band: the band is recomputed with the background shifted ±1σ and
  combined in quadrature with the statistical band. The observed limit uses the
  nominal background (no profiled nuisance parameters).
- **`profile`** — the systematic is added as `normsys` nuisance parameters
  profiled in the likelihood (affects observed *and* expected). `mode` selects the
  correlation structure: `components` (separate correlated NPs per ABCD component),
  `total` (single correlated NP), or `total_uncorrelated` (per-channel NP).

---

## Outputs

Written to `output_dir` with a prefix built from `label` plus tags for any active
transformations (e.g. `_sig8.98x`, `_bkgsyst`, `_obsrate…`, `_zerobkg`):

- `*_mu_2d_combined.png` (+ per region) — μ upper-limit colour map in (mass, ε[²])
- `*_mu_comparison.png` — μ=1 exclusion contours, combined + per category, with
  ±1σ/±2σ expected bands
- `*_mu_scan_per_region.png` — μ upper limit vs ε[²] per mass point
- `*_island_*.png` — CLs exclusion contours (when `run_cls: true`)
- `*_mu_contour_{obs,exp}.txt` — the μ=1 contour as `(mass, ε[²])` columns
- `*_summary_exclusion.png` — exclusion overlay (when `summary_plot: true`)
- `*_{background,signal}_yield_2d.png`, `*_sig_over_bkg_2d.png` — input diagnostics

---

## Example configs

| Config | Description |
|--------|-------------|
| `examples/config_z0.yaml` | SIMP z0-binned shape fit, unblinded 10% data, ABCD background systematics (expected-band) |
| `examples/config_z0_100pct.yaml` | Same, flat ×lumi projection to 100% |
| `examples/config_z0_100pct_obsrate.yaml` | 100% projection from observed-count rate (recommended projection) |
| `examples/config_ap_l1l1.yaml` | Nominal A′ displaced-vertex, ap_displaced, background-free |
| `examples/config_ap_l1l1_100pct.yaml` | Same, signal scaled to 100% |
| `examples/config_classic.yaml` | Classic single-channel cut-scan example |

---

## Programmatic API

The package can also be used directly (see `examples/example_limit.py`,
`json_limit.py`, `csv_limit.py`):

```python
from hps_limits import (
    load_z0_binned_cut_scan_jsons, load_results_z0_background_jsons,
    scan_cls_from_estimates, scan_mu_upper_from_estimates,
)
```

Key entry points: `scan_cls_from_estimates` (CLs island scan) and
`scan_mu_upper_from_estimates` (μ upper-limit scan); both take loaded
`BackgroundEstimate` / `SignalEstimate` objects and optional signal/background
systematics.
