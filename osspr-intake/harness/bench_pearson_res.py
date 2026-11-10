"""Time ``_calculate_res_sparse`` on a filtered single-cell count matrix.

The kernel walks every cell for every gene, including structural zeros.
The in-tree C++ path already replaces that with a nonzero scan plus a
closed form for zeros; the same rewrite inside this Numba function would
keep the residual variances and drop the genes-by-cells loop.

Prints one post-compilation wall time in milliseconds.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import scipy.sparse as sp

from omicverse.pp.experimental._highly_variable_genes import _calculate_res_sparse


def _counts(n_cells: int = 20000, n_genes: int = 10000, density: float = 0.04, seed: int = 0):
    """20k cells by 10k genes, 4% nonzero: a filtered scRNA-seq count matrix."""
    matrix = sp.random(
        n_cells,
        n_genes,
        density=density,
        format="csc",
        dtype=np.float64,
        random_state=seed,
    )
    matrix.data = np.ceil(np.abs(matrix.data) * 8.0)
    matrix.eliminate_zeros()
    sums_genes = np.asarray(matrix.sum(axis=0)).ravel()
    sums_cells = np.asarray(matrix.sum(axis=1)).ravel()
    return matrix, sums_genes, sums_cells


def _call(matrix, sums_genes, sums_cells):
    return _calculate_res_sparse(
        matrix.indptr,
        matrix.indices,
        matrix.data,
        sums_genes=sums_genes,
        sums_cells=sums_cells,
        sum_total=np.float64(sums_genes.sum()),
        clip=np.float64(np.sqrt(matrix.shape[0])),
        theta=np.float64(100.0),
        n_genes=matrix.shape[1],
        n_cells=matrix.shape[0],
    )


def main() -> None:
    matrix, sums_genes, sums_cells = _counts()
    warm = _call(matrix, sums_genes, sums_cells)  # compile once
    started = time.perf_counter()
    residuals = _call(matrix, sums_genes, sums_cells)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if residuals.shape != (matrix.shape[1],) or not np.isfinite(residuals).all():
        raise RuntimeError("residual variances are not a finite gene vector")
    if not np.isfinite(warm).all():
        raise RuntimeError("warmup residuals were not finite")
    print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
