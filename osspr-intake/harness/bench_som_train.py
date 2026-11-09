#!/usr/bin/env python3
"""Wall-time baseline for omicverse.flow._som.SOM.train.

FlowSOM's default batch is the full event matrix. Each epoch builds a
(n_events, n_nodes, n_markers) squared-distance temporary. A cytometry
file of a few tens of thousands of events on a 10x10 map is enough for
that temporary to dominate both time and the allocation.

Prints one elapsed time in milliseconds (one untimed warmup epoch-set
excluded via a separate SOM).
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

from omicverse.flow._som import SOM


def _events():
    rng = np.random.default_rng(0)
    # 30k events x 20 compensated markers. Three phenotypes plus noise,
    # the shape of a downsampled FCS file on a standard 10x10 map.
    n, n_markers = 30_000, 20
    centres = np.zeros((3, n_markers))
    centres[0, :6] = 8.0
    centres[1, 6:12] = 8.0
    centres[2, 12:] = 8.0
    which = rng.integers(0, 3, n)
    data = centres[which] + rng.normal(0.0, 0.8, size=(n, n_markers))
    return data.astype(np.float64)


def main() -> None:
    data = _events()
    SOM(grid=(10, 10), random_state=0).train(data, n_epochs=1)
    som = SOM(grid=(10, 10), random_state=0)
    t0 = time.perf_counter()
    som.train(data, n_epochs=6)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
