"""Time ``sepal`` on a synthetic square-lattice Visium crop.

Prints one steady-state wall time in milliseconds. Setup (coordinates,
counts, spatial graph) is excluded. A local change that diffuses every
gene with one batched stencil, instead of a Python loop over genes, can
cut this without changing scores.
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from anndata import AnnData

from omicverse.space._neighborhood import sepal
from omicverse.space._svg import spatial_neighbors


def _lattice(n_side: int = 40, n_genes: int = 48, seed: int = 0) -> AnnData:
    """40x40 grid (1,600 spots), 48 genes: a crop of a Visium slide plus a marker panel.

    One third of the genes are lifted inside the left half so the diffusion
    has a real spatial pattern to flatten, which is the case sepal is run for.
    """
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    counts = rng.poisson(2.0, size=(n_side * n_side, n_genes)).astype(np.float64)
    left = coords[:, 0] < (n_side / 2.0)
    counts[left, : n_genes // 3] += 15.0
    adata = AnnData(counts)
    adata.obsm["spatial"] = coords
    adata.var_names = [f"G{i}" for i in range(n_genes)]
    with contextlib.redirect_stdout(io.StringIO()):
        spatial_neighbors(adata, n_neighs=4, coord_type="grid")
    return adata


def main() -> None:
    adata = _lattice()
    # One untimed call settles allocator / import paths. The printed sample
    # is a single subsequent call, matching how the baseline median is taken.
    sepal(adata, max_neighs=4, copy=True)
    started = time.perf_counter()
    scores = sepal(adata, max_neighs=4, copy=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if scores is None or not np.isfinite(scores["sepal_score"].to_numpy()).any():
        raise RuntimeError("sepal returned no finite scores")
    print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
