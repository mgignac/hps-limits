#!/usr/bin/env python3
"""Convert ``*_eventcount.txt`` background-count tables into a background JSON
consumable by ``hps_limits.load_background_json`` (classic / ap_displaced mode).

Each input text file is one hit category and has a table with columns::

    Mass [MeV]   Before Cut   Poly Cut   Poly Eff   Zero Bkg Cut   Zero Bkg Eff

One of the integer count columns (``poly`` or ``zerobkg``) becomes the background
prediction ``bkg_est`` for that (region, mass).  The observation is set to the
same value (Asimov: ``n_a_obs = bkg_est``) since there is no real data here, and
the uncertainty is Poisson (``sqrt(N)``).

Example::

    python3.9 scripts/eventcount_to_background_json.py \\
        --inputs data/sarah-latest/rmng_bkg_0pt1/l1l1_eventcount.txt \\
                 data/sarah-latest/rmng_bkg_0pt1/l1l2_eventcount.txt \\
                 data/sarah-latest/rmng_bkg_0pt1/l2l2_eventcount.txt \\
        --column poly \\
        --output data/sarah-latest/rmng_bkg_0pt1/background_counts.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# token index of each count column in a numeric data row
# row: [mass, before_cut, poly_cut, poly_eff, zero_bkg_cut, zero_bkg_eff]
_COLUMN_INDEX = {"poly": 2, "zerobkg": 4}


def _region_from_filename(path: Path) -> str:
    """l1l1_eventcount.txt -> 'L1L1'."""
    return path.stem.split("_eventcount")[0].upper()


def _parse_counts(path: Path, col_index: int) -> list[tuple[float, int]]:
    """Return [(mass_MeV, count), ...] from one eventcount file."""
    rows: list[tuple[float, int]] = []
    for line in path.read_text().splitlines():
        toks = line.split()
        # data rows start with an integer mass and have all 6 columns
        if len(toks) >= 6 and toks[0].isdigit():
            rows.append((float(toks[0]), int(toks[col_index])))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="*_eventcount.txt files, one per hit category")
    ap.add_argument("--column", choices=sorted(_COLUMN_INDEX), default="poly",
                    help="which count column to use as the background (default: poly)")
    ap.add_argument("--output", required=True, help="output JSON path")
    ap.add_argument("--signal-window-gev", type=float, default=0.0,
                    help="placeholder signal_window_GeV (unused by the limit)")
    args = ap.parse_args()

    idx = _COLUMN_INDEX[args.column]
    regions: dict[str, list[dict]] = {}
    for f in args.inputs:
        p = Path(f)
        region = _region_from_filename(p)
        pts = []
        for mass_mev, count in _parse_counts(p, idx):
            err = math.sqrt(count) if count > 0 else 0.0
            pts.append({
                "mass_GeV": round(mass_mev / 1000.0, 6),
                "bkg_est": float(count),
                "bkg_est_err": err,
                "n_a_obs": float(count),      # Asimov: observed = predicted bkg
                "n_a_obs_err": err,
                "signal_window_GeV": args.signal_window_gev,
            })
        regions[region] = pts
        print(f"  {region}: {len(pts)} mass points, "
              f"total bkg = {sum(p['bkg_est'] for p in pts):.0f}")

    doc = {
        "meta": {
            "description": "Background counts from *_eventcount.txt",
            "column": args.column,
            "note": "n_a_obs set to bkg_est (Asimov); errors are Poisson sqrt(N)",
            "sources": [Path(f).name for f in args.inputs],
        },
        "regions": regions,
    }
    out = Path(args.output)
    out.write_text(json.dumps(doc, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
