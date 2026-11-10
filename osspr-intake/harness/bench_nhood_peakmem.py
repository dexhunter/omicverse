"""Peak Python heap of ``nhood_enrichment`` on a spatial atlas graph.

The function keeps every permutation's cluster-by-cluster count, an array
of shape ``(n_perms, n_types, n_types)``, then reduces it to a mean and a
standard deviation. An online (Welford) reduction inside the same loop
uses the same shuffles and returns the same z-scores without that tensor.

Prints tracemalloc's peak bytes for allocations made during the call.
The AnnData and the spatial graph are built before tracing starts.
"""
from __future__ import annotations

import contextlib
import io
import sys
import tracemalloc
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from anndata import AnnData

from omicverse.space._neighborhood import nhood_enrichment
from omicverse.space._svg import spatial_neighbors


def _atlas(n_spots: int = 10000, n_types: int = 100, n_neighs: int = 6, seed: int = 0) -> AnnData:
    """10k spots, 100 cell types, 6 spatial neighbours: a detailed atlas on one slide."""
    rng = np.random.default_rng(seed)
    adata = AnnData(np.zeros((n_spots, 2), dtype=np.float32))
    adata.obsm["spatial"] = rng.random((n_spots, 2))
    adata.obs["cell_type"] = pd.Categorical(
        [f"type{i:03d}" for i in rng.integers(0, n_types, n_spots)]
    )
    with contextlib.redirect_stdout(io.StringIO()):
        spatial_neighbors(adata, n_neighs=n_neighs)
    return adata


def main() -> None:
    adata = _atlas()
    tracemalloc.start()
    result = nhood_enrichment(
        adata, "cell_type", n_perms=1000, seed=0, copy=True
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    zscore = result["zscore"]
    if zscore.shape != (100, 100) or not np.isfinite(zscore).all():
        raise RuntimeError("nhood_enrichment did not return a finite 100x100 z-score")
    print(str(int(peak)))


if __name__ == "__main__":
    main()
