#!/usr/bin/env python3
"""Wall-time baseline for omicverse.es._aucell._func_aucell.

The CPU kernel ranks every gene of every cell with a Python call to
scipy.stats.rankdata, then keeps only ranks inside the top n_up for the
recovery curve. This harness times that kernel on a dense count-like
matrix of the size one AUCell chunk sees.

Prints one elapsed time in milliseconds (numba warmup excluded).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from omicverse.es._aucell import _func_aucell


def _problem(n_cells: int = 4000, n_genes: int = 3000, n_sig: int = 40, genes_per: int = 50):
    rng = np.random.default_rng(0)
    # Discrete counts: ties are the common case, matching scRNA input.
    mat = rng.integers(0, 12, size=(n_cells, n_genes)).astype(np.float64)
    cnct = np.empty(n_sig * genes_per, dtype=np.int64)
    for s in range(n_sig):
        cnct[s * genes_per : (s + 1) * genes_per] = rng.choice(
            n_genes, genes_per, replace=False
        )
    offsets = np.full(n_sig, genes_per, dtype=np.int64)
    starts = np.zeros(n_sig, dtype=np.int64)
    starts[1:] = np.cumsum(offsets)[:-1]
    # Default AUCell window is the top 5% of genes.
    n_up = int(np.ceil(0.05 * n_genes))
    return mat, cnct, starts, offsets, n_up


def main() -> None:
    mat, cnct, starts, offsets, n_up = _problem()
    _func_aucell(mat[:32], cnct, starts, offsets, n_up=n_up, verbose=False)
    t0 = time.perf_counter()
    _func_aucell(mat, cnct, starts, offsets, n_up=n_up, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
