#!/usr/bin/env python3
"""Peak-memory baseline for omicverse.space.co_occurrence.

co_occurrence materialises scipy.spatial.distance_matrix for every pair
of spots, then a boolean mask of pairs inside each radius. On a
Visium-sized section that dense n x n block is the allocation.

Prints tracemalloc peak bytes for one call (data construction excluded).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import gc
import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import anndata as ad
import numpy as np
import pandas as pd

from omicverse.space._neighborhood import co_occurrence


def _slide(n: int = 4000):
    rng = np.random.default_rng(0)
    # Tissue-shaped coordinates, eight cell types. n=4000 is one Visium
    # capture area; the same call on a Visium HD binning is larger.
    coords = np.column_stack(
        [
            rng.uniform(0.0, 6000.0, n),
            rng.uniform(0.0, 6000.0, n),
        ]
    )
    adata = ad.AnnData(np.zeros((n, 4), dtype=np.float32))
    adata.obsm["spatial"] = coords
    adata.obs["cell_type"] = pd.Categorical(rng.choice(
        [f"c{i}" for i in range(8)], n
    ))
    return adata


def main() -> None:
    adata = _slide()
    gc.collect()
    tracemalloc.start()
    co_occurrence(adata, "cell_type", interval=6, copy=True)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(int(peak))


if __name__ == "__main__":
    main()
