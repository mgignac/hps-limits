#!/usr/bin/env python3
"""Plot Zbi in the epsilon vs mass plane from a scan results JSON file."""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_data(path):
    with open(path) as f:
        obj = json.load(f)
    return obj["metadata"], obj["data"]


def make_grid(data, masses, epsilons):
    mass_idx = {m: i for i, m in enumerate(masses)}
    eps_idx = {e: i for i, e in enumerate(epsilons)}
    grid = np.full((len(epsilons), len(masses)), np.nan)
    for pt in data:
        i = eps_idx.get(pt["epsilon"])
        j = mass_idx.get(pt["mass_MeV"])
        if i is not None and j is not None:
            grid[i, j] = pt["Zbi"]
    return grid


def plot_zbi(path, outdir, tag):
    meta, data = load_data(path)

    masses = meta["mass_grid_MeV"]
    epsilons = meta["epsilon_grid"]

    grid = make_grid(data, masses, epsilons)

    fig, ax = plt.subplots(figsize=(8, 6))

    mass_arr = np.array(masses)
    eps_arr = np.array(epsilons)

    # pcolormesh expects bin edges; use midpoint-based edges
    def edges(arr):
        log = np.log10(arr) if arr[0] > 0 and (arr[-1] / arr[0]) > 10 else None
        if log is not None:
            c = np.concatenate([[log[0] - (log[1] - log[0]) / 2],
                                 (log[:-1] + log[1:]) / 2,
                                 [log[-1] + (log[-1] - log[-2]) / 2]])
            return 10 ** c
        else:
            c = np.concatenate([[arr[0] - (arr[1] - arr[0]) / 2],
                                 (arr[:-1] + arr[1:]) / 2,
                                 [arr[-1] + (arr[-1] - arr[-2]) / 2]])
            return c

    mass_edges = edges(mass_arr)
    eps_edges = edges(eps_arr)

    pcm = ax.pcolormesh(mass_edges, eps_edges, grid, cmap="viridis", shading="flat")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$Z_\mathrm{bi}$", fontsize=13)

    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{A'}$ [MeV]", fontsize=13)
    ax.set_ylabel(r"$\varepsilon$", fontsize=13)

    suffix = meta.get("suffix", "")
    minz0 = meta.get("min_z0_cut", "")
    projsig = meta.get("proj_sig_cut", "")
    title_parts = [r"$Z_\mathrm{bi}$ scan"]
    if suffix:
        title_parts.append(suffix)
    if minz0:
        title_parts.append(f"min z0={minz0}")
    if projsig:
        title_parts.append(f"proj sig={projsig}")
    ax.set_title("  |  ".join(title_parts), fontsize=11)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    fname = f"{base}_{tag}_zbi.pdf" if tag else f"{base}_zbi.pdf"
    outpath = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(outpath)
    print(f"Saved: {outpath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Zbi in the epsilon vs mass plane.")
    parser.add_argument("file", help="Path to results JSON file")
    parser.add_argument("-o", "--outdir", default="plots", help="Output directory (default: plots)")
    parser.add_argument("-t", "--tag", default="", help="Tag appended to output filename")
    args = parser.parse_args()

    plot_zbi(args.file, args.outdir, args.tag)


if __name__ == "__main__":
    main()
