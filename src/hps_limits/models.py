"""pyhf model construction for HPS search results."""

from __future__ import annotations

import pyhf
import numpy as np


def build_model(
    signal: float | list[float],
    background: float | list[float],
    background_uncertainty: float | list[float],
) -> tuple[pyhf.Model, list[float]]:
    """Build a pyhf model for an HPS search result.

    Parameters
    ----------
    signal:
        Expected signal yield(s). A single float for one channel,
        or a list for multiple channels (e.g. run periods or mass windows).
    background:
        Expected background yield(s), matching the shape of ``signal``.
    background_uncertainty:
        Absolute background uncertainty (1σ) for each channel.

    Returns
    -------
    model:
        A pyhf Model with an uncorrelated background modifier.
    data:
        Observed data placeholder set equal to the background expectation.
        Replace with real observed counts before running limits.
    """
    signal = np.atleast_1d(np.asarray(signal, dtype=float)).tolist()
    background = np.atleast_1d(np.asarray(background, dtype=float)).tolist()
    background_uncertainty = np.atleast_1d(
        np.asarray(background_uncertainty, dtype=float)
    ).tolist()

    if not (len(signal) == len(background) == len(background_uncertainty)):
        raise ValueError(
            "signal, background, and background_uncertainty must have the same length"
        )

    model = pyhf.simplemodels.uncorrelated_background(
        signal=signal,
        bkg=background,
        bkg_uncertainty=background_uncertainty,
    )
    # Default observed = background (Asimov-like placeholder)
    observed = background + model.config.auxdata
    return model, observed


def build_workspace(
    signal: float | list[float],
    background: float | list[float],
    background_uncertainty: float | list[float],
    observed: float | list[float] | None = None,
) -> dict:
    """Build a pyhf workspace dictionary for an HPS search result.

    The workspace JSON can be serialised and used with the pyhf CLI or
    shared as a likelihood publication.

    Parameters
    ----------
    signal, background, background_uncertainty:
        Same as :func:`build_model`.
    observed:
        Observed data counts. Defaults to background (Asimov).

    Returns
    -------
    dict
        A workspace dictionary following the pyhf schema.
    """
    signal = np.atleast_1d(np.asarray(signal, dtype=float)).tolist()
    background = np.atleast_1d(np.asarray(background, dtype=float)).tolist()
    background_uncertainty = np.atleast_1d(
        np.asarray(background_uncertainty, dtype=float)
    ).tolist()

    if observed is None:
        observed = background
    observed = np.atleast_1d(np.asarray(observed, dtype=float)).tolist()

    n_bins = len(signal)
    rel_uncertainty = [u / b for u, b in zip(background_uncertainty, background)]

    workspace = {
        "channels": [
            {
                "name": "search",
                "samples": [
                    {
                        "name": "signal",
                        "data": signal,
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": "background",
                        "data": background,
                        "modifiers": [
                            {
                                "name": f"bkg_unc_bin{i}",
                                "type": "shapesys",
                                "data": [u if j == i else 0.0 for j in range(n_bins)],
                            }
                            for i in range(n_bins)
                        ],
                    },
                ],
            }
        ],
        "observations": [{"name": "search", "data": observed}],
        "measurements": [
            {
                "name": "HPS",
                "config": {
                    "poi": "mu",
                    "parameters": [],
                },
            }
        ],
        "version": "1.0.0",
    }
    return workspace
