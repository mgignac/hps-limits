#!/usr/bin/env python3
"""HPS limits driver — run CLs / μ upper-limit scans from a YAML config file.

Usage::

    python3.9 scripts/run_limits.py --config examples/config_z0.yaml
    python3.9 scripts/run_limits.py --config examples/config_classic.yaml

Any YAML key can be overridden on the command line::

    python3.9 scripts/run_limits.py --config examples/config_z0.yaml \\
        --bkg-zero-floor --no-per-region --mu-cap 5000

Requires: pyyaml  (pip install pyyaml  or  pip install hps-limits[dev])
"""

from __future__ import annotations

import argparse
import copy
import math
import re
import sys
from collections import defaultdict
from itertools import cycle
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional imports — give a clear error if missing
# ---------------------------------------------------------------------------

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install pyyaml")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

import numpy as np

from hps_limits import (
    broaden_mu_band_with_systematic,
    load_ap_signal_json,
    load_ap_signal_jsons,
    load_background_json,
    load_background_systematics_json,
    load_cut_scan_jsons,
    load_results_json,
    load_results_z0_background_jsons,
    load_z0_binned_cut_scan_jsons,
    scan_cls_from_estimates,
    scan_mu_upper_from_estimates,
    synthetic_zero_background,
)
from hps_limits.plotting import (
    extract_mu_contour,
    load_exclusion_txt,
    plot_cls_band,
    plot_cls_contour,
    plot_island_exclusion,
    plot_mu_upper_2d,
    plot_mu_upper_band,
    plot_mu_upper_contour,
    plot_mu_upper_scan,
    plot_summary_exclusion,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BKG_FLOOR_VALUE     = 100.0
_BKG_FLOOR_THRESHOLD = 1e-6   # z0-binned loader clamps true zeros to 1e-10

_PALETTE = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]

# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "label":                  "HPS",
    "mode":                   "z0",
    "output_dir":             "plots",
    "simps":                  False,
    "per_region":             True,
    "run_cls":                True,
    "run_mu":                 True,
    "mu_cap":                 10000.0,
    "cl_level":               0.95,
    "bkg_zero_floor":         False,
    # z0-binned inputs
    "z0_files":               [],
    # separate background for z0 mode (glob patterns / folder supported)
    "background_jsons":       [],    # if set, background loaded from here instead of z0_files
    "background_projection":  None,  # e.g. "10% projection"
    "background_z0_labels":   None,  # override bin labels (default: ["lo","hi"] for 2-bin)
    # classic inputs
    "background_json":        None,
    "signal_files":           [],
    # ap_displaced signal format (used when signal_format = "ap_displaced")
    "signal_format":          "cut_scan",   # "cut_scan" or "ap_displaced"
    "signal_json":            None,         # single-file ap_displaced (legacy)
    "signal_region":          "L1L1",       # region label for single-file mode
    "signal_jsons":           [],           # multi-file ap_displaced: [{file, region}, ...]
    # post-load transformations
    "signal_scale":           1.0,    # multiply all signal yields by this factor
    # per-region signal scale, e.g. {L1L1: 1.4, L1L2: 1.5}; applied after signal_scale
    "signal_scale_by_region": None,
    "background_scale":       1.0,    # multiply all background + observed yields by this factor
    # mass-axis translation (e.g. SIMP m_VD → m_A): multiply the mass of each
    # signal/background entry by this factor before matching (1.0 = no change).
    "signal_mass_scale":      1.0,
    "background_mass_scale":  1.0,
    # project background from measured observed counts: {lumi_ratio: <float>}
    "background_observed_rate": None,
    "zero_background":        False,  # replace all backgrounds + observed with ~0
    # set observed = N per channel (excess on ~zero background); None = off
    "assume_observed":        None,
    # signal systematics (list of {name, value} dicts; value is relative 1σ)
    "signal_systematics":      [],
    # background systematics (ABCD method): {file, mode} dict, or None to disable.
    # mode ∈ {"components", "total", "total_uncorrelated"}.
    "background_systematics":  None,
    # plot annotations (ROOT-style, drawn on the μ 2D maps)
    "experiment_label":       None,   # e.g. "HPS work in progress"
    "lumi_label":             None,   # e.g. "18.01 pb$^{-1}$ @ 3.74 GeV"
    "y_max":                  None,   # truncate the ε²(or ε) axis at this value
    "plot_format":            "png",  # output image format for saved plots (png|pdf|…)
    # summary plot
    "interpolate_contour_mass": False,  # smooth contour across sparse mass grid
    "summary_plot":           False,  # produce a summary exclusion overlay plot
    "prior_exclusions":       [],     # list of {label, file, color, linestyle, linewidth}
}

_REQUIRED_BY_MODE = {
    "z0":      ["z0_files"],
    "classic": [],   # background_json optional when zero_background + ap_displaced
}


def _resolve(raw: str | Path, base: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base / p).resolve()


def _expand_paths(raw_list: list, base: Path) -> list[Path]:
    """Resolve a list of path strings (globs allowed) relative to *base*."""
    result: list[Path] = []
    for raw in raw_list:
        p = _resolve(raw, base)
        if "*" in str(p) or "?" in str(p):
            expanded = sorted(p.parent.glob(p.name))
            if not expanded:
                sys.exit(f"Config error: glob matched nothing: {raw}")
            result.extend(expanded)
        else:
            result.append(p)
    return result


def _expand_paths_or_dir(raw_list: list, base: Path) -> list[Path]:
    """Like _expand_paths but also accepts a bare directory — expands to *.json."""
    result: list[Path] = []
    for raw in raw_list:
        p = _resolve(raw, base)
        if p.is_dir():
            expanded = sorted(p.glob("*.json"))
            if not expanded:
                sys.exit(f"Config error: no *.json files found in directory: {p}")
            result.extend(expanded)
        elif "*" in str(p) or "?" in str(p):
            expanded = sorted(p.parent.glob(p.name))
            if not expanded:
                sys.exit(f"Config error: glob matched nothing: {raw}")
            result.extend(expanded)
        else:
            result.append(p)
    return result


def load_config(config_path: Path, overrides: dict) -> dict:
    """Load YAML config, apply defaults, apply CLI overrides, validate."""
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = dict(_DEFAULTS)
    cfg.update(raw)
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    base = config_path.parent

    # Resolve output directory
    cfg["output_dir"] = _resolve(cfg["output_dir"], base)

    # Resolve input paths
    mode = cfg["mode"]
    if mode not in ("z0", "classic"):
        sys.exit(f"Config error: mode must be 'z0' or 'classic', got '{mode}'")

    missing = [k for k in _REQUIRED_BY_MODE[mode] if not cfg.get(k)]
    if missing:
        sys.exit(f"Config error: mode '{mode}' requires: {missing}")

    if mode == "z0":
        cfg["z0_files"] = _expand_paths(cfg["z0_files"], base)
        if cfg["background_jsons"]:
            cfg["background_jsons"] = _expand_paths_or_dir(
                cfg["background_jsons"], base
            )
    else:
        sig_fmt = cfg.get("signal_format", "cut_scan")
        if sig_fmt not in ("cut_scan", "ap_displaced"):
            sys.exit(f"Config error: signal_format must be 'cut_scan' or "
                     f"'ap_displaced', got '{sig_fmt}'")
        if sig_fmt == "cut_scan":
            if not cfg.get("background_json"):
                sys.exit("Config error: mode 'classic' with signal_format "
                         "'cut_scan' requires 'background_json'")
            cfg["background_json"] = _resolve(cfg["background_json"], base)
            if not cfg.get("signal_files"):
                sys.exit("Config error: mode 'classic' with signal_format "
                         "'cut_scan' requires 'signal_files'")
            cfg["signal_files"] = _expand_paths(cfg["signal_files"], base)
        else:
            # ap_displaced: background_json required unless zero_background
            if cfg.get("background_json"):
                cfg["background_json"] = _resolve(cfg["background_json"], base)
            elif not cfg.get("zero_background"):
                sys.exit("Config error: signal_format 'ap_displaced' requires "
                         "'background_json' (or set zero_background: true)")
            # Signal: multi-file (signal_jsons) or single-file (signal_json)
            if cfg.get("signal_jsons"):
                resolved = []
                for entry in cfg["signal_jsons"]:
                    if not isinstance(entry, dict) or "file" not in entry or "region" not in entry:
                        sys.exit("Config error: each signal_jsons entry must have 'file' and 'region'")
                    resolved.append({
                        "file":   _resolve(entry["file"], base),
                        "region": entry["region"],
                    })
                cfg["signal_jsons"] = resolved
            elif cfg.get("signal_json"):
                cfg["signal_json"] = _resolve(cfg["signal_json"], base)
            else:
                sys.exit("Config error: signal_format 'ap_displaced' requires "
                         "'signal_json' or 'signal_jsons'")

    # Resolve prior exclusion file paths
    for entry in cfg.get("prior_exclusions", []):
        if "file" in entry:
            entry["file"] = str(_resolve(entry["file"], base))

    # Validate background_observed_rate (mutually exclusive with background_scale)
    obs_rate = cfg.get("background_observed_rate")
    if obs_rate:
        if not isinstance(obs_rate, dict) or "lumi_ratio" not in obs_rate:
            sys.exit("Config error: background_observed_rate must be a mapping "
                     "with a 'lumi_ratio' key.")
        obs_rate["lumi_ratio"] = float(obs_rate["lumi_ratio"])
        if cfg.get("background_scale", 1.0) != 1.0:
            sys.exit("Config error: set either background_observed_rate or "
                     "background_scale, not both.")

    # Resolve background-systematics file path and validate mode
    bsyst = cfg.get("background_systematics")
    if bsyst:
        if not isinstance(bsyst, dict) or "file" not in bsyst:
            sys.exit("Config error: background_systematics must be a mapping "
                     "with a 'file' key (and optional 'mode').")
        bsyst["file"] = _resolve(bsyst["file"], base)
        mode = bsyst.get("mode", "components")
        if mode not in ("components", "total", "total_uncorrelated"):
            sys.exit(f"Config error: background_systematics mode must be "
                     f"'components', 'total', or 'total_uncorrelated', got '{mode}'")
        bsyst["mode"] = mode
        treatment = bsyst.get("treatment", "profile")
        if treatment not in ("profile", "expected_band"):
            sys.exit(f"Config error: background_systematics treatment must be "
                     f"'profile' or 'expected_band', got '{treatment}'")
        bsyst["treatment"] = treatment

    return cfg


def _output_prefix(cfg: dict) -> str:
    """Sanitised label for use in output filenames."""
    label = re.sub(r"[^\w]+", "_", cfg["label"]).strip("_").lower()
    parts = []
    if cfg["bkg_zero_floor"]:
        parts.append("bkgfloor100")
    if cfg["zero_background"]:
        parts.append("zerobkg")
    if cfg.get("assume_observed") is not None:
        parts.append(f"obs{cfg['assume_observed']:g}")
    scale = cfg.get("signal_scale", 1.0)
    if scale != 1.0:
        parts.append(f"sig{scale:g}x")
    bscale = cfg.get("background_scale", 1.0)
    if bscale != 1.0:
        parts.append(f"bkg{bscale:g}x")
    obs_rate = cfg.get("background_observed_rate")
    if obs_rate:
        parts.append(f"obsrate{obs_rate['lumi_ratio']:g}x")
    if cfg.get("background_systematics"):
        parts.append("bkgsyst")
    suffix = ("_" + "_".join(parts)) if parts else ""
    return label + suffix


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_inputs(cfg: dict):
    """Load background and signal from the paths specified in *cfg*."""
    if cfg["mode"] == "z0":
        bg_from_sig_files, sig = load_z0_binned_cut_scan_jsons(cfg["z0_files"])
        if cfg["background_jsons"]:
            proj = cfg.get("background_projection")
            bg = load_results_z0_background_jsons(
                cfg["background_jsons"],
                projection=proj,
                bin_labels=cfg.get("background_z0_labels"),
            )
        else:
            bg = bg_from_sig_files  # old-style: background embedded in signal files
    elif cfg.get("signal_format") == "ap_displaced":
        if cfg.get("signal_jsons"):
            sig = load_ap_signal_jsons(
                [(e["file"], e["region"]) for e in cfg["signal_jsons"]]
            )
        else:
            sig = load_ap_signal_json(
                cfg["signal_json"],
                region=cfg.get("signal_region", "L1L1"),
            )
        if cfg.get("background_json"):
            bg = load_background_json(
                cfg["background_json"],
                projection=cfg.get("background_projection"),
            )
        else:
            bg = synthetic_zero_background(sig)
    else:
        bg, _ = load_results_json(
            cfg["background_json"],
            projection=cfg.get("background_projection"),
        )
        _, sig = load_cut_scan_jsons(cfg["signal_files"])
    return bg, sig


# ---------------------------------------------------------------------------
# Background floor
# ---------------------------------------------------------------------------


def apply_floor(bg, enabled: bool) -> int:
    """Replace near-zero backgrounds with _BKG_FLOOR_VALUE.  Returns count."""
    if not enabled:
        return 0
    n = 0
    for e in bg.entries:
        if e.nbkg < _BKG_FLOOR_THRESHOLD:
            e.nbkg     = _BKG_FLOOR_VALUE
            e.nbkg_err = math.sqrt(_BKG_FLOOR_VALUE)
            e.observed = _BKG_FLOOR_VALUE
            n += 1
    return n



def apply_signal_scale(sig, scale: float) -> None:
    """Multiply all signal yields by *scale* in-place."""
    if scale == 1.0:
        return
    for e in sig.entries:
        e.nsig *= scale


def apply_signal_scale_by_region(sig, scale_map: dict) -> None:
    """Multiply each region's signal yields by a per-region factor, in-place.

    ``scale_map`` maps region name → multiplicative factor (e.g.
    ``{"L1L1": 1.4, "L1L2": 1.5, "L2L2": 1.45}``).  Regions absent from the map
    are left unchanged.
    """
    for e in sig.entries:
        f = float(scale_map.get(e.region, 1.0))
        if f != 1.0:
            e.nsig *= f


def _shifted_background_by_syst(bg, syst_map: dict, sign: float):
    """Return a deep copy of *bg* with each channel scaled by (1 + sign·σ_total).

    σ_total is the per-channel ``total_frac`` from the background-systematics
    map; channels with no entry are left unchanged.  Used to build the ±1σ
    background variations for the explicit expected-band treatment.
    """
    bg2 = copy.deepcopy(bg)
    for e in bg2.entries:
        frac = float((syst_map.get(e.region) or {}).get("total_frac", 0.0))
        scale = 1.0 + sign * frac
        e.nbkg = max(e.nbkg * scale, 0.0)
        e.nbkg_err *= scale
        e.observed = e.observed * scale
        e.observed_err *= scale
    return bg2


def apply_background_observed_rate(bg, lumi_ratio: float) -> int:
    """Project the background from the measured observed counts to full lumi.

    For each (region/z0-bin, mass) the background prediction *and* the (Asimov)
    observation are set to ``n_a_obs × lumi_ratio`` — i.e. the measured
    event rate scaled to the full dataset.  Bins with **zero** observed events
    fall back to the ABCD estimate (``bkg_est × lumi_ratio``) so none collapses
    to zero.  Errors are Poisson on the projected count.

    There is no real full-luminosity data, so the observation is set equal to
    the projected background (Asimov); the "observed" limit therefore coincides
    with the median expected.  Returns the number of zero-observed fallback bins.
    """
    n_fallback = 0
    for e in bg.entries:
        n_obs = float(e.observed)          # n_a_obs at the measured luminosity
        if n_obs > 0.0:
            proj = n_obs * lumi_ratio
        else:
            proj = float(e.nbkg) * lumi_ratio   # ABCD bkg_est fallback
            n_fallback += 1
        e.nbkg = proj
        e.observed = proj
        e.nbkg_err = math.sqrt(proj) if proj > 0.0 else 0.0
        e.observed_err = math.sqrt(proj) if proj > 0.0 else 0.0
    return n_fallback


def apply_background_scale(bg, scale: float) -> None:
    """Multiply all background and observed yields by *scale* in-place.

    Scales the central values (``nbkg``, ``observed``) and their absolute
    uncertainties (``nbkg_err``, ``observed_err``) by the same factor, which
    preserves the fractional uncertainty — appropriate for a flat
    luminosity-ratio projection of a data-driven estimate.
    """
    if scale == 1.0:
        return
    for e in bg.entries:
        e.nbkg         *= scale
        e.nbkg_err     *= scale
        e.observed     *= scale
        e.observed_err *= scale


def apply_zero_background(bg) -> None:
    """Set all background yields and observed counts to zero in-place.

    ``_pyhf_spec_from_entries`` clamps nbkg ≥ 1e-10, so this effectively
    gives a background-free model.  Setting observed = 0 ensures the
    likelihood is evaluated at zero observed events.
    """
    for e in bg.entries:
        e.nbkg     = 0.0
        e.nbkg_err = 0.0
        e.observed = 0.0


def apply_assume_observed(bg, n_obs: float) -> None:
    """Set the observed count to *n_obs* in every channel, in-place.

    The background prediction is left untouched (≈0 for a zero-background
    search), so the assumed events are an observed excess on top of it.  Used to
    study how the observed limit degrades if N events are seen.
    """
    n = float(n_obs)
    err = math.sqrt(n) if n > 0.0 else 0.0
    for e in bg.entries:
        e.observed = n
        e.observed_err = err


# ---------------------------------------------------------------------------
# Diagnostic printout
# ---------------------------------------------------------------------------


def print_data_summary(bg, sig, cfg: dict) -> None:
    coupling_label = "ε" if cfg["simps"] else "ε²"
    W = 72

    bg_masses = sorted({e.mass_mev for e in bg.entries})
    n_floor   = sum(1 for e in bg.entries if e.nbkg < _BKG_FLOOR_THRESHOLD)
    n_small   = sum(1 for e in bg.entries
                    if _BKG_FLOOR_THRESHOLD <= e.nbkg < 1.0)

    print("\n" + "=" * W)
    print("BACKGROUND ESTIMATES")
    print("=" * W)
    print(f"  {len(bg.entries)} entries  |  {len(bg.regions)} regions  |  "
          f"{len(bg_masses)} mass points")
    floor_note = ("  [will be floored]" if cfg["bkg_zero_floor"]
                  else "  [pass bkg_zero_floor: true to floor these]")
    print(f"  Effectively zero (nbkg < {_BKG_FLOOR_THRESHOLD:.0e}): "
          f"{n_floor} entries{floor_note}")
    print(f"  Near-zero (< 1 event):                   {n_small} entries")

    for region in bg.regions:
        r_entries = sorted(
            [e for e in bg.entries if e.region == region],
            key=lambda e: e.mass_mev,
        )
        n_floor_r  = sum(1 for e in r_entries if e.nbkg < _BKG_FLOOR_THRESHOLD)
        floored_tag = (f"  [{n_floor_r} floored]"    if cfg["bkg_zero_floor"] and n_floor_r else
                       f"  [{n_floor_r} will floor]" if n_floor_r else "")
        print(f"\n  {region}{floored_tag}")
        print(f"    {'Mass/MeV':>9}  {'Nbkg':>12}  {'±err':>12}  {'Observed':>10}")
        print(f"    {'-'*9}  {'-'*12}  {'-'*12}  {'-'*10}")
        for e in r_entries:
            flag = ("  <-- zero" if e.nbkg < _BKG_FLOOR_THRESHOLD else
                    "  <-- <1"  if e.nbkg < 1.0 else "")
            print(f"    {e.mass_mev:>9.1f}  {e.nbkg:>12.4g}  "
                  f"{e.nbkg_err:>12.4g}  {e.observed:>10.4g}{flag}")

    print("\n" + "=" * W)
    print("SIGNAL ESTIMATES")
    print("=" * W)
    print(f"  {len(sig.entries)} entries  |  {len(sig.regions)} regions  |  "
          f"{len(sig.masses_mev)} mass points  |  "
          f"{len(sig.eps2_values)} {coupling_label} points")

    col = (f"{'Region':<14}  {'Masses':>6}  {coupling_label+' min':>12}  "
           f"{coupling_label+' max':>12}  {'Nsig min':>10}  {'Nsig max':>10}")
    print(f"\n  {col}")
    print(f"  {'-'*14}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}")
    for r in sig.regions:
        r_ent     = [e for e in sig.entries if e.region == r]
        masses_r  = sorted({e.mass_mev for e in r_ent})
        eps2_vals = [e.eps2 for e in r_ent]
        nsig_vals = [e.nsig for e in r_ent]
        e2_lo = (math.sqrt(min(eps2_vals)) if cfg["simps"] else min(eps2_vals))
        e2_hi = (math.sqrt(max(eps2_vals)) if cfg["simps"] else max(eps2_vals))
        print(f"  {r:<14}  {len(masses_r):>6}  {e2_lo:>12.3e}  {e2_hi:>12.3e}  "
              f"{min(nsig_vals):>10.4f}  {max(nsig_vals):>10.4f}")

    print("=" * W)


# ---------------------------------------------------------------------------
# Per-region category grouping
# ---------------------------------------------------------------------------


def build_categories(bg, sig, mode: str) -> dict[str, list[str]]:
    """Return {hit_category: [channel_names]} for per-region scans.

    In z0-binned mode channels are named ``{hit_cat}_{bin}`` (e.g. L1L1_A);
    they are grouped so each category run uses all its z0 bins together.
    """
    common = set(bg.regions) & set(sig.regions)

    if mode == "z0":
        groups: dict[str, list[str]] = defaultdict(list)
        for r in bg.regions:
            if r not in common:
                continue
            parts = r.rsplit("_", 1)
            cat = parts[0] if len(parts) == 2 and len(parts[1]) <= 2 else r
            groups[cat].append(r)
        seen: dict[str, None] = {}
        for r in bg.regions:
            parts = r.rsplit("_", 1)
            cat = parts[0] if len(parts) == 2 and len(parts[1]) <= 2 else r
            if cat in groups:
                seen[cat] = None
        return {cat: groups[cat] for cat in seen}
    else:
        return {r: [r] for r in bg.regions if r in common}


# ---------------------------------------------------------------------------
# CLs analysis
# ---------------------------------------------------------------------------


def run_cls_analysis(cfg: dict, bg, sig, categories, colors, prefix: str) -> None:
    out_dir    = cfg["output_dir"]
    eps_axis   = cfg["simps"]
    per_region = cfg["per_region"]
    cl_level   = cfg["cl_level"]
    eps2_range = (min(sig.eps2_values), max(sig.eps2_values))
    label      = cfg["label"]
    fmt        = cfg.get("plot_format", "png")

    _fmt = lambda v: f"{math.sqrt(v):.3e}" if eps_axis else f"{v:.3e}"

    print(f"\n{'='*60}\nRunning CLs scan (CL={cl_level*100:.0f}%) ...\n{'='*60}")

    # ── Combined ────────────────────────────────────────────────────────────
    print("\n  [Combined]")
    sig_systs = cfg.get("signal_systematics") or None
    bkg_systs = cfg.get("_bkg_syst_map")
    bkg_mode  = (cfg.get("background_systematics") or {}).get("mode", "components")
    treatment = (cfg.get("background_systematics") or {}).get("treatment", "profile")
    # Only profile the bkg systematics as NPs when treatment == "profile".
    # For "expected_band" they are applied post-hoc to the μ band, not here.
    profile_systs = bkg_systs if treatment == "profile" else None
    cls_results = scan_cls_from_estimates(
        bg, sig, signal_systematics=sig_systs,
        background_systematics=profile_systs, bkg_syst_mode=bkg_mode,
        cl_level=cl_level, verbose=True,
    )
    print(f"  → {len(cls_results)} mass points")

    coupling_label = "ε" if eps_axis else "ε²"
    col_w = max(len(f"Obs {coupling_label} upper"), 16)
    print(f"\n{'Mass':>8}  {'Obs '+coupling_label+' upper':>{col_w}}  "
          f"{'Exp '+coupling_label+' upper':>{col_w}}")
    print("-" * (8 + 2 * col_w + 4))
    for sr in cls_results:
        obs = sr.limit.eps2_upper_obs
        exp = sr.limit.eps2_upper_exp
        print(f"{sr.mass:>8.0f}  "
              f"{_fmt(obs) if obs and not math.isnan(obs) else '---':>{col_w}}  "
              f"{_fmt(exp) if exp and not math.isnan(exp) else '---':>{col_w}}")

    cl_pct = int(cl_level * 100)
    _coup  = np.sqrt if eps_axis else lambda x: x
    ylabel = (rf"$\varepsilon$ ({cl_pct}% CL exclusion)" if eps_axis
              else rf"$\varepsilon^2$ ({cl_pct}% CL exclusion)")
    interp = cfg["interpolate_contour_mass"]

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_cls_band(cls_results, ax=ax, cl_level=cl_level,
                  eps_axis=eps_axis, interpolate_mass=interp)
    plot_cls_contour(cls_results, ax=ax, which="exp", cl_level=cl_level,
                     eps_axis=eps_axis, interpolate_mass=interp,
                     colors=["black"], linewidths=[1.6], linestyles=["--"])
    plot_cls_contour(cls_results, ax=ax, which="obs", cl_level=cl_level,
                     eps_axis=eps_axis, interpolate_mass=interp,
                     colors=["black"], linewidths=[2.2])
    ax.set_xlabel(r"$m_{A'}$ (MeV)")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_ylim(_coup(eps2_range[0]), _coup(eps2_range[1]))
    ax.set_title(f"{label} — Exclusion Contour ({cl_pct}% CL, Combined)")
    from matplotlib.lines import Line2D as _L2D
    from matplotlib.patches import Patch as _Patch
    ax.legend(handles=[
        _L2D([0],[0], color="black", linestyle="-",  linewidth=2.2, label="Observed"),
        _L2D([0],[0], color="black", linestyle="--", linewidth=1.6, label="Expected"),
        _Patch(facecolor="limegreen", label=r"Exp. $\pm 1\sigma$"),
        _Patch(facecolor="gold", label=r"Exp. $\pm 2\sigma$"),
    ])
    fig.tight_layout()
    out = out_dir / f"{prefix}_island_combined.{fmt}"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    plt.close(fig)

    if not per_region:
        return

    # ── Per-region ──────────────────────────────────────────────────────────
    per_region_cls: dict[str, list] = {}
    for cat, chans in categories.items():
        print(f"\n  [{cat}]  channels={chans}")
        per_region_cls[cat] = scan_cls_from_estimates(
            bg, sig, select_regions=chans, signal_systematics=sig_systs,
            background_systematics=profile_systs, bkg_syst_mode=bkg_mode,
            cl_level=cl_level, verbose=False,
        )
        print(f"  → {len(per_region_cls[cat])} mass points")

        fig_r, ax_r = plt.subplots(figsize=(10, 5))
        plot_cls_band(per_region_cls[cat], ax=ax_r, cl_level=cl_level,
                      eps_axis=eps_axis, interpolate_mass=interp)
        plot_cls_contour(per_region_cls[cat], ax=ax_r, which="exp",
                         cl_level=cl_level, eps_axis=eps_axis, interpolate_mass=interp,
                         colors=["black"], linewidths=[1.6], linestyles=["--"])
        plot_cls_contour(per_region_cls[cat], ax=ax_r, which="obs",
                         cl_level=cl_level, eps_axis=eps_axis, interpolate_mass=interp,
                         colors=["black"], linewidths=[2.2])
        ax_r.set_xlabel(r"$m_{A'}$ (MeV)")
        ax_r.set_ylabel(ylabel)
        ax_r.set_yscale("log")
        ax_r.set_ylim(_coup(eps2_range[0]), _coup(eps2_range[1]))
        ax_r.set_title(f"{label} — Exclusion Contour ({cl_pct}% CL, {cat})")
        from matplotlib.patches import Patch as _PatchR
        ax_r.legend(handles=[
            Line2D([0],[0], color="black", linestyle="-",  linewidth=2.2, label="Observed"),
            Line2D([0],[0], color="black", linestyle="--", linewidth=1.6, label="Expected"),
            _PatchR(facecolor="limegreen", label=r"Exp. $\pm 1\sigma$"),
            _PatchR(facecolor="gold", label=r"Exp. $\pm 2\sigma$"),
        ])
        fig_r.tight_layout()
        out_r = out_dir / f"{prefix}_island_{cat}.{fmt}"
        fig_r.savefig(out_r, dpi=150)
        print(f"  Saved {out_r}")
        plt.close(fig_r)

    # Overlay comparison
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 5))
    legend_handles = []
    # Combined ±1σ/±2σ expected band (drawn first, behind the lines)
    plot_cls_band(cls_results, ax=ax_cmp, cl_level=cl_level,
                  eps_axis=eps_axis, interpolate_mass=interp)
    from matplotlib.patches import Patch as _PatchC
    legend_handles += [
        _PatchC(facecolor="gold", label=r"Exp. $\pm 2\sigma$"),
        _PatchC(facecolor="limegreen", label=r"Exp. $\pm 1\sigma$"),
    ]
    for cat, results_r in per_region_cls.items():
        col = colors[cat]
        plot_cls_contour(results_r, ax=ax_cmp, which="obs", cl_level=cl_level,
                         eps_axis=eps_axis, interpolate_mass=interp,
                         colors=[col], linewidths=[1.2], linestyles=["-"], alpha=0.85)
        legend_handles.append(
            Line2D([0],[0], color=col, linestyle="-", linewidth=1.2, label=cat)
        )
    plot_cls_contour(cls_results, ax=ax_cmp, which="exp", cl_level=cl_level,
                     eps_axis=eps_axis, interpolate_mass=interp,
                     colors=["black"], linewidths=[1.6], linestyles=["--"])
    plot_cls_contour(cls_results, ax=ax_cmp, which="obs", cl_level=cl_level,
                     eps_axis=eps_axis, interpolate_mass=interp,
                     colors=["black"], linewidths=[2.2])
    legend_handles += [
        Line2D([0],[0], color="black", linestyle="--", linewidth=1.6,
               label="Combined (exp.)"),
        Line2D([0],[0], color="black", linestyle="-",  linewidth=2.2,
               label="Combined (obs.)"),
    ]
    ax_cmp.set_xlabel(r"$m_{A'}$ (MeV)")
    ax_cmp.set_ylabel(ylabel)
    ax_cmp.set_yscale("log")
    ax_cmp.set_ylim(_coup(eps2_range[0]), _coup(eps2_range[1]))
    ax_cmp.set_title(f"{label} — Region Comparison: Exclusion Contour ({cl_pct}% CL)")
    ax_cmp.legend(handles=legend_handles, fontsize=10, ncol=2)
    fig_cmp.tight_layout()
    out_cmp = out_dir / f"{prefix}_island_comparison.{fmt}"
    fig_cmp.savefig(out_cmp, dpi=150)
    print(f"\nSaved {out_cmp}")
    plt.close(fig_cmp)


# ---------------------------------------------------------------------------
# μ analysis
# ---------------------------------------------------------------------------


def run_mu_analysis(cfg: dict, bg, sig, categories, colors, prefix: str) -> None:
    out_dir    = cfg["output_dir"]
    eps_axis   = cfg["simps"]
    per_region = cfg["per_region"]
    mu_cap     = cfg["mu_cap"]
    cl_level   = cfg["cl_level"]
    eps2_range = (min(sig.eps2_values), max(sig.eps2_values))
    label      = cfg["label"]
    _coup      = np.sqrt if eps_axis else lambda x: x

    print(f"\n{'='*60}\nRunning μ upper-limit scan "
          f"(CL={cl_level*100:.0f}%, cap={mu_cap:.0f}) ...\n"
          f"This may take several minutes.\n{'='*60}")

    # ── Combined ────────────────────────────────────────────────────────────
    print("\n  [Combined]")
    sig_systs = cfg.get("signal_systematics") or None
    bkg_systs = cfg.get("_bkg_syst_map")
    bkg_mode  = (cfg.get("background_systematics") or {}).get("mode", "components")
    treatment = (cfg.get("background_systematics") or {}).get("treatment", "profile")
    profile_systs = bkg_systs if treatment == "profile" else None
    mu_results = scan_mu_upper_from_estimates(
        bg, sig, signal_systematics=sig_systs,
        background_systematics=profile_systs, bkg_syst_mode=bkg_mode,
        cl_level=cl_level, mu_upper_cap=mu_cap, verbose=True,
    )
    print(f"  → {len(mu_results)} mass points")

    # Explicit expected-band treatment: broaden the combined expected band with
    # the background systematic, computed from ±1σ background variations.
    # Observed limit and per-region results are left at nominal background.
    if treatment == "expected_band" and bkg_systs:
        print("\n  [background-systematics] Computing ±1σ background variations "
              "for the expected band ...")
        bg_up = _shifted_background_by_syst(bg, bkg_systs, +1.0)
        bg_dn = _shifted_background_by_syst(bg, bkg_systs, -1.0)
        mu_up = scan_mu_upper_from_estimates(
            bg_up, sig, signal_systematics=sig_systs,
            cl_level=cl_level, mu_upper_cap=mu_cap, verbose=False,
        )
        mu_dn = scan_mu_upper_from_estimates(
            bg_dn, sig, signal_systematics=sig_systs,
            cl_level=cl_level, mu_upper_cap=mu_cap, verbose=False,
        )
        broaden_mu_band_with_systematic(mu_results, mu_up, mu_dn)
        print("  → expected band broadened (stat ⊕ ABCD systematic, in quadrature)")

    exp_lbl  = cfg.get("experiment_label")
    lumi_lbl = cfg.get("lumi_label")
    y_max    = cfg.get("y_max")
    fmt      = cfg.get("plot_format", "png")
    # Observed and expected on separate single-panel canvases (one file each)
    for which in ("obs", "exp"):
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        plot_mu_upper_2d(mu_results, ax=ax2, which=which, eps_axis=eps_axis,
                         cl_level=cl_level, experiment_label=exp_lbl,
                         lumi_label=lumi_lbl, y_max=y_max)
        fig2.tight_layout()
        out2 = out_dir / f"{prefix}_mu_2d_combined_{which}.{fmt}"
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"Saved {out2}")
        plt.close(fig2)

    if not per_region:
        _save_1d_scan(mu_results, out_dir, prefix, label, eps_axis, fmt, cl_level)
        _write_contour_txt(mu_results, out_dir, prefix, cfg)
        _write_mu_grid_json(mu_results, None, out_dir, prefix, cfg)
        _maybe_save_summary(cfg, mu_results, out_dir, prefix)
        return

    # ── Per-region ──────────────────────────────────────────────────────────
    per_region_mu: dict[str, list] = {}
    for cat, chans in categories.items():
        print(f"\n  [{cat}]  channels={chans}")
        per_region_mu[cat] = scan_mu_upper_from_estimates(
            bg, sig, select_regions=chans, signal_systematics=sig_systs,
            background_systematics=profile_systs, bkg_syst_mode=bkg_mode,
            cl_level=cl_level, mu_upper_cap=mu_cap, verbose=False,
        )
        print(f"  → {len(per_region_mu[cat])} mass points")

        for which in ("obs", "exp"):
            fig_r2, ax_r2 = plt.subplots(figsize=(8, 6))
            plot_mu_upper_2d(per_region_mu[cat], ax=ax_r2, which=which,
                             eps_axis=eps_axis, cl_level=cl_level,
                             experiment_label=exp_lbl, lumi_label=lumi_lbl, y_max=y_max)
            fig_r2.tight_layout()
            out_r2 = out_dir / f"{prefix}_mu_2d_{cat}_{which}.{fmt}"
            fig_r2.savefig(out_r2, dpi=150, bbox_inches="tight")
            print(f"  Saved {out_r2}")
            plt.close(fig_r2)

    # Contour overlay comparison — observed and expected on separate canvases
    interp = cfg["interpolate_contour_mass"]
    cl_str = f"{int(cl_level*100)}% CL"
    ylabel = (rf"$\varepsilon$ ({cl_str}, $\mu=1$ contour)" if eps_axis
              else rf"$\varepsilon^2$ ({cl_str}, $\mu=1$ contour)")

    for which in ("exp", "obs"):
        fig_cmp2, ax_cmp2 = plt.subplots(figsize=(10, 5))
        legend_handles = []
        ls = "--" if which == "exp" else "-"

        # ±1σ/±2σ expected band (expected canvas only; drawn behind the lines)
        if which == "exp":
            plot_mu_upper_band(mu_results, ax=ax_cmp2, eps_axis=eps_axis)
            legend_handles += [
                Line2D([0], [0], color="gold",      linewidth=6, alpha=0.55, label=r"Exp. $\pm 2\sigma$"),
                Line2D([0], [0], color="limegreen", linewidth=6, alpha=0.70, label=r"Exp. $\pm 1\sigma$"),
            ]

        for cat, results_r in per_region_mu.items():
            col = colors[cat]
            plot_mu_upper_contour(results_r, ax=ax_cmp2, which=which,
                                  eps_axis=eps_axis, interpolate_mass=interp,
                                  colors=[col], linewidths=[1.4], linestyles=[ls])
            legend_handles.append(
                Line2D([0], [0], color=col, linestyle=ls, linewidth=1.4, label=cat)
            )
        plot_mu_upper_contour(mu_results, ax=ax_cmp2, which=which,
                              eps_axis=eps_axis, interpolate_mass=interp,
                              colors=["black"], linewidths=[2.0], linestyles=[ls])
        _wl = "exp." if which == "exp" else "obs."
        legend_handles.append(
            Line2D([0], [0], color="black", linestyle=ls, linewidth=2.0,
                   label=f"Combined ({_wl})")
        )
        ax_cmp2.set_xlabel(r"$m_{A'}$ (MeV)")
        ax_cmp2.set_ylabel(ylabel)
        ax_cmp2.set_yscale("log")
        if y_max is not None:
            ax_cmp2.set_ylim(top=float(y_max))
        _wt = "Expected" if which == "exp" else "Observed"
        ax_cmp2.set_title(f"{label} — Region Comparison ({_wt}): μ=1 Exclusion Contour ({cl_str})")
        ax_cmp2.legend(handles=legend_handles, fontsize=11, ncol=2, loc="upper right")
        fig_cmp2.tight_layout()
        out_cmp2 = out_dir / f"{prefix}_mu_comparison_{which}.{fmt}"
        fig_cmp2.savefig(out_cmp2, dpi=150)
        print(f"\nSaved {out_cmp2}")
        plt.close(fig_cmp2)

    # Per-mass 1D panels
    _save_1d_scan_per_region(mu_results, per_region_mu, colors, out_dir,
                             prefix, label, eps_axis, fmt, cl_level)

    _write_contour_txt(mu_results, out_dir, prefix, cfg)
    _write_mu_grid_json(mu_results, per_region_mu, out_dir, prefix, cfg)
    _maybe_save_summary(cfg, mu_results, out_dir, prefix)


def _write_contour_txt(mu_results, out_dir, prefix, cfg) -> None:
    """Write the μ=1 exclusion contour (obs + exp) to two-column text files."""
    eps_axis = cfg["simps"]
    col2 = "eps" if eps_axis else "eps2"
    label = cfg["label"]

    for which in ("obs", "exp"):
        segs = extract_mu_contour(mu_results, which=which, level=1.0,
                                  eps_axis=eps_axis)
        if not segs:
            print(f"  [contour] No μ=1 contour found ({which}); skipping text file.")
            continue

        out = out_dir / f"{prefix}_mu_contour_{which}.txt"
        cl_pct = int(round(cfg["cl_level"] * 100))
        with open(out, "w") as fh:
            fh.write(f"# {label} — μ=1 exclusion contour ({which}, {cl_pct}% CL)\n")
            fh.write(f"# mass_MeV  {col2}\n")
            for k, seg in enumerate(segs):
                if len(segs) > 1:
                    fh.write(f"# --- segment {k + 1} ---\n")
                for mass, coup in seg:
                    fh.write(f"{mass:.6g}  {coup:.6e}\n")
        print(f"  Saved {out}")


def _write_mu_grid_json(mu_results, per_region_mu, out_dir, prefix, cfg) -> None:
    """Dump the full μ upper-limit grid (obs + expected band) to JSON.

    One record per mass, each with the ε² grid and the observed / median-expected
    μ upper limits plus the 5-point expected band (−2σ, −1σ, median, +1σ, +2σ).
    Includes the combined result and, if available, each hit category.
    """
    import json

    def _pack(results):
        recs = []
        for r in results:
            recs.append({
                "mass_MeV": float(r.mass),
                "eps2":        [float(x) for x in r.eps2_arr],
                "mu_obs":      [float(x) for x in r.mu_upper_obs_arr],
                "mu_exp":      [float(x) for x in r.mu_upper_exp_arr],
                "mu_exp_band": [[float(v) for v in row]
                                for row in r.mu_upper_exp_band_arr],
            })
        return recs

    doc = {
        "meta": {
            "label":      cfg["label"],
            "cl_level":   cfg["cl_level"],
            "coupling":   "eps" if cfg["simps"] else "eps2",
            "mu_cap":     cfg["mu_cap"],
            "band_order": ["-2sigma", "-1sigma", "median", "+1sigma", "+2sigma"],
            "note":       "mu = signal-strength upper limit; mu<=1 is excluded",
        },
        "combined": _pack(mu_results),
    }
    if per_region_mu:
        doc["regions"] = {cat: _pack(res) for cat, res in per_region_mu.items()}

    out = out_dir / f"{prefix}_mu_grid.json"
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  Saved {out}")


def _maybe_save_summary(cfg, mu_results, out_dir, prefix):
    if not cfg.get("summary_plot"):
        return

    eps_axis = cfg["simps"]
    label    = cfg["label"]
    prior    = cfg.get("prior_exclusions", [])
    fmt      = cfg.get("plot_format", "png")

    fig, ax = plt.subplots(figsize=(10, 6))

    cl_str   = f"{int(round(cfg['cl_level'] * 100))}% CL"
    new_lbl_obs = f"{label} (obs., {cl_str})"
    new_lbl_exp = f"{label} (exp., {cl_str})"

    plot_summary_exclusion(
        mu_results, prior_curves=prior, ax=ax,
        which="exp", eps_axis=eps_axis,
        new_label=new_lbl_exp,
        new_color="black", new_linestyle="--", new_linewidth=1.8,
    )
    plot_summary_exclusion(
        mu_results, prior_curves=None, ax=ax,
        which="obs", eps_axis=eps_axis,
        new_label=new_lbl_obs,
        new_color="black", new_linestyle="-", new_linewidth=2.2,
    )

    coupling_tex = (rf"$\varepsilon$ ({cl_str}, $\mu=1$ contour)"
                    if eps_axis
                    else rf"$\varepsilon^2$ ({cl_str}, $\mu=1$ contour)")
    ax.set_ylabel(coupling_tex)
    ax.set_title(f"{label} — Summary Exclusion ({cl_str})")

    # Rebuild legend to merge prior + new entries with no duplicates
    handles, labels_l = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, l in zip(handles, labels_l):
        if l not in seen:
            seen[l] = h
    if seen:
        ax.legend(handles=list(seen.values()), labels=list(seen.keys()),
                  fontsize=10, loc="upper right")

    fig.tight_layout()
    out = out_dir / f"{prefix}_summary_exclusion.{fmt}"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    plt.close(fig)


def _save_1d_scan(mu_results, out_dir, prefix, label, eps_axis, fmt="png",
                  cl_level=0.95):
    n = len(mu_results)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    for idx, mr in enumerate(mu_results):
        plot_mu_upper_scan(mr, ax=axes[idx // ncols][idx % ncols],
                           eps_axis=eps_axis, cl_level=cl_level)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    coup_tex = r"$\varepsilon$" if eps_axis else r"$\varepsilon^2$"
    fig.suptitle(rf"{label} — $\mu$ upper limit vs {coup_tex}", y=1.01)
    fig.tight_layout()
    out = out_dir / f"{prefix}_mu_scan.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def _save_1d_scan_per_region(mu_results, per_region_mu, colors,
                              out_dir, prefix, label, eps_axis, fmt="png",
                              cl_level=0.95):
    _coup = np.sqrt if eps_axis else lambda x: x
    coup_tex = r"$\varepsilon$" if eps_axis else r"$\varepsilon^2$"

    def _draw(ax, mr_comb, legend_fs=8):
        mass = mr_comb.mass
        # Combined result with ±1σ/±2σ band (drawn first, behind per-region lines)
        plot_mu_upper_scan(mr_comb, ax=ax, show_band=True, eps_axis=eps_axis,
                           cl_level=cl_level)
        # Per-region expected lines overlaid (no band, just lines)
        for cat, results_r in per_region_mu.items():
            mr_r = next((m for m in results_r if m.mass == mass), None)
            if mr_r is None:
                continue
            ax.plot(_coup(mr_r.eps2_arr), mr_r.mu_upper_exp_arr,
                    color=colors[cat], linestyle="--", linewidth=1.2,
                    alpha=0.85, label=cat)
        ax.set_title(rf"$m_{{A'}}$ = {mass:.0f} MeV")
        ax.legend(fontsize=legend_fs, ncol=2)

    # ── Grid overview (all masses on one canvas) ──────────────────────────────
    n = len(mu_results)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    for idx, mr_comb in enumerate(mu_results):
        _draw(axes[idx // ncols][idx % ncols], mr_comb, legend_fs=6)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(rf"{label} — $\mu$ upper limit vs {coup_tex} (per region)",
                 y=1.01)
    fig.tight_layout()
    out = out_dir / f"{prefix}_mu_scan_per_region.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

    # ── Individual per-mass canvases (one file per mass) ──────────────────────
    for mr_comb in mu_results:
        figi, axi = plt.subplots(figsize=(7, 5))
        _draw(axi, mr_comb)
        figi.tight_layout()
        outi = out_dir / f"{prefix}_mu_scan_m{mr_comb.mass:.0f}.{fmt}"
        figi.savefig(outi, dpi=150, bbox_inches="tight")
        print(f"  Saved {outi}")
        plt.close(figi)


# ---------------------------------------------------------------------------
# Yield diagnostic plots
# ---------------------------------------------------------------------------


def _save_yield_plots(bg, sig, cfg: dict, out_dir, prefix) -> None:
    """Save 2D colour-map plots of background and peak signal yields."""
    import matplotlib.colors as mcolors

    label    = cfg["label"]
    eps_axis = cfg["simps"]
    fmt      = cfg.get("plot_format", "png")
    _coup    = np.sqrt if eps_axis else lambda x: x
    coup_tex = r"$\varepsilon$" if eps_axis else r"$\varepsilon^2$"

    regions  = bg.regions
    masses   = sorted({e.mass_mev for e in bg.entries})

    # ── Background 2D ────────────────────────────────────────────────────────
    bg_grid = np.full((len(regions), len(masses)), np.nan)
    bg_by_reg = bg.by_region()
    for ri, reg in enumerate(regions):
        for ei in bg_by_reg.get(reg, []):
            mi = masses.index(ei.mass_mev)
            bg_grid[ri, mi] = ei.nbkg

    fig_b, ax_b = plt.subplots(figsize=(14, max(4, len(regions) * 0.6 + 1)))
    finite_b = bg_grid[np.isfinite(bg_grid) & (bg_grid > 0)]
    if len(finite_b):
        vmin_b = max(float(np.percentile(finite_b, 2)), 1e-2)
        vmax_b = max(float(np.percentile(finite_b, 98)), vmin_b * 10)
        norm_b = mcolors.LogNorm(vmin=vmin_b, vmax=vmax_b)
    else:
        norm_b = None   # all-zero background — linear scale
    im_b = ax_b.pcolormesh(
        masses, np.arange(len(regions)),
        bg_grid,
        norm=norm_b,
        cmap="viridis", shading="nearest",
    )
    fig_b.colorbar(im_b, ax=ax_b, label="Background yield (events)")
    ax_b.set_yticks(range(len(regions)))
    ax_b.set_yticklabels(regions, fontsize=8)
    ax_b.set_xlabel(r"$m_{A'}$ (MeV)")
    ax_b.set_title(f"{label} — Background yield per region")
    fig_b.tight_layout()
    out_b = out_dir / f"{prefix}_background_yield_2d.{fmt}"
    fig_b.savefig(out_b, dpi=150, bbox_inches="tight")
    print(f"Saved {out_b}")
    plt.close(fig_b)

    # ── Peak signal 2D ───────────────────────────────────────────────────────
    sig_regions = sig.regions
    sig_masses  = sig.masses_mev
    sig_by_reg  = sig.by_region()

    peak_grid = np.full((len(sig_regions), len(sig_masses)), np.nan)
    peak_eps2 = np.full((len(sig_regions), len(sig_masses)), np.nan)

    for ri, reg in enumerate(sig_regions):
        entries_r = sig_by_reg.get(reg, [])
        by_mass: dict[float, list] = {}
        for e in entries_r:
            by_mass.setdefault(e.mass_mev, []).append(e)
        for mi, m in enumerate(sig_masses):
            pts = by_mass.get(m, [])
            if pts:
                best = max(pts, key=lambda e: e.nsig)
                peak_grid[ri, mi] = best.nsig
                peak_eps2[ri, mi] = best.eps2

    fig_s, ax_s = plt.subplots(figsize=(14, max(4, len(sig_regions) * 0.6 + 1)))
    finite_s = peak_grid[np.isfinite(peak_grid) & (peak_grid > 0)]
    vmin_s = float(np.percentile(finite_s, 2)) if len(finite_s) else 1e-4
    vmax_s = float(np.percentile(finite_s, 98)) if len(finite_s) else 10.0
    im_s = ax_s.pcolormesh(
        sig_masses, np.arange(len(sig_regions)),
        peak_grid,
        norm=mcolors.LogNorm(vmin=max(vmin_s, 1e-10), vmax=vmax_s),
        cmap="plasma", shading="nearest",
    )
    fig_s.colorbar(im_s, ax=ax_s, label=f"Peak signal yield (events, at optimum {coup_tex})")
    ax_s.set_yticks(range(len(sig_regions)))
    ax_s.set_yticklabels(sig_regions, fontsize=8)
    ax_s.set_xlabel(r"$m_{A'}$ (MeV)")
    ax_s.set_title(f"{label} — Peak signal yield per region")
    fig_s.tight_layout()
    out_s = out_dir / f"{prefix}_signal_yield_2d.{fmt}"
    fig_s.savefig(out_s, dpi=150, bbox_inches="tight")
    print(f"Saved {out_s}")
    plt.close(fig_s)

    # ── Signal / Background ratio 2D ─────────────────────────────────────────
    common_regs = [r for r in sig_regions if r in regions]
    common_mass = sorted(set(masses) & set(sig_masses))
    if common_regs and common_mass:
        ratio_grid = np.full((len(common_regs), len(common_mass)), np.nan)
        bg_lookup  = {(e.region, e.mass_mev): e.nbkg for e in bg.entries}
        for ri, reg in enumerate(common_regs):
            for mi, m in enumerate(common_mass):
                pts = [e for e in sig_by_reg.get(reg, []) if e.mass_mev == m]
                if pts:
                    nsig_peak = max(e.nsig for e in pts)
                    nbkg = bg_lookup.get((reg, m), np.nan)
                    if nbkg > 0 and nsig_peak > 0:
                        ratio_grid[ri, mi] = nsig_peak / nbkg

        finite_r = ratio_grid[np.isfinite(ratio_grid) & (ratio_grid > 0)]
        if len(finite_r):
            fig_r, ax_r = plt.subplots(figsize=(14, max(4, len(common_regs) * 0.6 + 1)))
            vmin_r = float(np.percentile(finite_r, 2))
            vmax_r = float(np.percentile(finite_r, 98))
            im_r = ax_r.pcolormesh(
                common_mass, np.arange(len(common_regs)),
                ratio_grid,
                norm=mcolors.LogNorm(vmin=max(vmin_r, 1e-10), vmax=max(vmax_r, 1e-9)),
                cmap="RdYlGn", shading="nearest",
            )
            fig_r.colorbar(im_r, ax=ax_r, label=f"Peak signal / background")
            ax_r.set_yticks(range(len(common_regs)))
            ax_r.set_yticklabels(common_regs, fontsize=8)
            ax_r.set_xlabel(r"$m_{A'}$ (MeV)")
            ax_r.set_title(f"{label} — Peak signal / background ratio")
            fig_r.tight_layout()
            out_r = out_dir / f"{prefix}_sig_over_bkg_2d.{fmt}"
            fig_r.savefig(out_r, dpi=150, bbox_inches="tight")
            print(f"Saved {out_r}")
            plt.close(fig_r)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, metavar="YAML",
                        help="Path to YAML configuration file")
    # Optional CLI overrides (None = don't override)
    parser.add_argument("--cls-only",      action="store_true", default=None,
                        help="Override: run CLs scan only")
    parser.add_argument("--mu-only",       action="store_true", default=None,
                        help="Override: run μ scan only")
    parser.add_argument("--no-per-region", action="store_true", default=None,
                        help="Override: skip per-region breakdown")
    parser.add_argument("--bkg-zero-floor",action="store_true", default=None,
                        help="Override: floor zero backgrounds at 100 events")
    parser.add_argument("--mu-cap",        type=float, default=None,
                        help="Override: cap on reported μ upper limit")
    parser.add_argument("--max-mass-points", type=int, default=None, metavar="N",
                        help="Only use the first N mass points (for quick tests)")
    args = parser.parse_args()

    # Build override dict (only populated when explicitly passed)
    overrides: dict = {}
    if args.cls_only:
        overrides["run_mu"] = False
    if args.mu_only:
        overrides["run_cls"] = False
    if args.no_per_region:
        overrides["per_region"] = False
    if args.bkg_zero_floor:
        overrides["bkg_zero_floor"] = True
    if args.mu_cap is not None:
        overrides["mu_cap"] = args.mu_cap

    cfg = load_config(Path(args.config), overrides)

    print(f"Config : {args.config}")
    print(f"Mode   : {cfg['mode']}")
    print(f"Label  : {cfg['label']}")
    print(f"Output : {cfg['output_dir']}")
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    # Load data
    bg, sig = load_inputs(cfg)

    # Mass-axis translation (e.g. SIMP m_VD → m_A).  Scale each entry's mass
    # before signal/background are matched, so the whole analysis (matching,
    # contours, plots) is expressed on the translated axis.
    sms = float(cfg.get("signal_mass_scale", 1.0) or 1.0)
    bms = float(cfg.get("background_mass_scale", 1.0) or 1.0)
    if sms != 1.0:
        for e in sig.entries:
            e.mass_mev = round(e.mass_mev * sms, 6)
    if bms != 1.0:
        for e in bg.entries:
            e.mass_mev = round(e.mass_mev * bms, 6)
    if sms != 1.0 or bms != 1.0:
        common = sorted(set(e.mass_mev for e in sig.entries)
                        & set(e.mass_mev for e in bg.entries))
        print(f"\n[mass-scale] signal × {sms:g}, background × {bms:g}  → "
              f"{len(common)} common masses: {common}")

    # Load background systematics (ABCD method), if configured
    cfg["_bkg_syst_map"] = None
    bsyst = cfg.get("background_systematics")
    if bsyst:
        cfg["_bkg_syst_map"] = load_background_systematics_json(
            bsyst["file"], bin_labels=cfg.get("background_z0_labels")
        )
        print(f"\n[background-systematics] Loaded "
              f"{len(cfg['_bkg_syst_map'])} channel(s) from {bsyst['file']}"
              f"  (mode={bsyst['mode']})")
        for ch, e in cfg["_bkg_syst_map"].items():
            comps = ", ".join(f"{k}={v*100:.2f}%"
                              for k, v in e["components_frac"].items())
            print(f"    {ch:<12} total={e['total_frac']*100:.3f}%  [{comps}]")

    if args.max_mass_points is not None:
        keep = set(sorted({e.mass_mev for e in sig.entries})[:args.max_mass_points])
        sig.entries = [e for e in sig.entries if e.mass_mev in keep]
        bg.entries  = [e for e in bg.entries  if e.mass_mev in keep]
        print(f"\n[max-mass-points] Keeping {len(keep)} mass point(s): "
              f"{sorted(keep)}")

    # Diagnostic printout on raw (pre-floor) data
    print_data_summary(bg, sig, cfg)

    # Apply floor
    n_floored = apply_floor(bg, cfg["bkg_zero_floor"])
    if cfg["bkg_zero_floor"]:
        print(f"\n[bkg-zero-floor] Applied: {n_floored} entries set to "
              f"{_BKG_FLOOR_VALUE:.0f} events "
              f"(nbkg_err = {math.sqrt(_BKG_FLOOR_VALUE):.1f})")

    # Zero background
    if cfg["zero_background"]:
        apply_zero_background(bg)
        print(f"\n[zero-background] All {len(bg.entries)} background entries "
              f"set to 0 (observed = 0)")

    # Assume a fixed observed count per channel (excess on ~zero background)
    if cfg.get("assume_observed") is not None:
        apply_assume_observed(bg, cfg["assume_observed"])
        print(f"\n[assume-observed] Observed set to {cfg['assume_observed']:g} "
              f"in all {len(bg.entries)} channels (background unchanged)")

    # Background from observed-count rate (projection), else flat scale
    obs_rate = cfg.get("background_observed_rate")
    if obs_rate:
        ratio = float(obs_rate["lumi_ratio"])
        n_fb = apply_background_observed_rate(bg, ratio)
        print(f"\n[background-observed-rate] Background projected from observed "
              f"counts × {ratio:g} (Asimov: obs = bkg).  {n_fb} zero-observed "
              f"bin(s) fell back to ABCD bkg_est × {ratio:g}")
    else:
        bscale = cfg.get("background_scale", 1.0)
        if bscale != 1.0:
            apply_background_scale(bg, bscale)
            print(f"\n[background-scale] All background + observed yields "
                  f"multiplied by {bscale:g}")

    # Signal scale
    scale = cfg.get("signal_scale", 1.0)
    if scale != 1.0:
        apply_signal_scale(sig, scale)
        print(f"\n[signal-scale] All signal yields multiplied by {scale:g}")

    # Per-region signal scale (applied on top of signal_scale)
    sbr = cfg.get("signal_scale_by_region")
    if sbr:
        apply_signal_scale_by_region(sig, sbr)
        print(f"\n[signal-scale-by-region] Per-category factors: "
              + ", ".join(f"{k}×{v:g}" for k, v in sbr.items()))

    n_grid = len(sig.masses_mev) * len(sig.eps2_values)
    coupling_label = "ε" if cfg["simps"] else "ε²"
    print(f"\nGrid size: {len(sig.masses_mev)} masses × "
          f"{len(sig.eps2_values)} {coupling_label} points = {n_grid} total")

    categories = build_categories(bg, sig, cfg["mode"])
    print(f"Region categories: {list(categories)}")

    _color_cycle = cycle(_PALETTE)
    colors = {cat: next(_color_cycle) for cat in categories}

    prefix = _output_prefix(cfg)

    _save_yield_plots(bg, sig, cfg, cfg["output_dir"], prefix)

    if cfg["run_cls"]:
        run_cls_analysis(cfg, bg, sig, categories, colors, prefix)

    if cfg["run_mu"]:
        run_mu_analysis(cfg, bg, sig, categories, colors, prefix)


if __name__ == "__main__":
    main()
