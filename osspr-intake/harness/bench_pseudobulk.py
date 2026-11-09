#!/usr/bin/env python3
"""Wall-time baseline for omicverse.single.pseudobulk.

Realistic cohort aggregation: sparse counts, many donor x cell-type
profiles, and sample-level metadata columns that the function copies
onto profiles only when they are constant within a profile. The
constancy check is a Python loop over profiles.

Prints one elapsed time in milliseconds (warmup excluded) to stdout.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import contextlib
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import anndata as ad
import numpy as np
import scipy.sparse as sp

from omicverse.single._pseudobulk import pseudobulk


def _adata():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 30_000, 800
    nnz = int(n_cells * n_genes * 0.10)
    rows = rng.integers(0, n_cells, nnz)
    cols = rng.integers(0, n_genes, nnz)
    data = rng.integers(1, 12, nnz).astype(np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes))
    adata = ad.AnnData(X=X)
    n_donors, n_types = 200, 40
    donors = np.array([f"d{i}" for i in range(n_donors)])
    types = np.array([f"t{i}" for i in range(n_types)])
    adata.obs["donor"] = rng.choice(donors, n_cells)
    adata.obs["celltype"] = rng.choice(types, n_cells)
    # Clinical fields constant within a donor, the usual columns
    # pseudobulk is asked to carry onto each profile.
    for k in range(12):
        donor_val = {d: f"v{k}_{i % 7}" for i, d in enumerate(donors)}
        adata.obs[f"meta{k}"] = adata.obs["donor"].map(donor_val)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    return adata


def main() -> None:
    adata = _adata()
    with contextlib.redirect_stdout(io.StringIO()):
        pseudobulk(adata, sample_col="donor", groups_col="celltype", mode="sum")
        t0 = time.perf_counter()
        pseudobulk(adata, sample_col="donor", groups_col="celltype", mode="sum")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
