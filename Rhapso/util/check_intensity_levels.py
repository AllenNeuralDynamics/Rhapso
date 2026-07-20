#!/usr/bin/env python3
"""
Intensity statistics for AIND NGFF/Zarr hybrid stores.

This script:
  • Loads a multiscale tile from SPIM.ome.zarr using the SAME logic as your pipeline
  • Uses s3fs.S3Map (required for AIND virtualized Zarr stores)
  • Loads level N (default: 0)
  • Computes robust intensity statistics

Works even though AIND uses Zarr-v3-style chunk hashing.
"""

import zarr
import numpy as np
import s3fs
import dask.array as da

# -------------------------------------------------------------
# USER PARAMS
# -------------------------------------------------------------
BUCKET_TILE_PATH = (
    "aind-open-data/exaSPIM_802450_2025-11-25_16-55-46_processed_2025-12-02_15-17-21/flatfield_correction/SPIM.ome.zarr/tile_000000_ch_488.zarr"
)

MULTISCALE_LEVEL = 4   # choose 0–6 (0 = full-res)

ANON = True            # If authentication required, set ANON = False
# -------------------------------------------------------------


def open_tile_group(tile_path: str, anon: bool = True):
    """
    Opens the AIND Zarr tile using the same technique as your pipeline:
      • Uses S3Map
      • Opens as a group (not an array)
      • Lets zarr/dask resolve v3 chunk layout
    """
    s3 = s3fs.S3FileSystem(anon=anon)

    # REQUIRED: use S3Map, NOT SimpleStore
    store = s3fs.S3Map(root=tile_path, s3=s3)

    # open as a group (root of the tile)
    g = zarr.open(store, mode="r")
    return g


def load_multiscale_level(group, level: int):
    """
    Returns dask array for multiscale level:
      Dimensions: (t, c, z, y, x)
    """
    # group[level] is a *group*, so da.from_zarr loads the underlying array
    arr = da.from_zarr(group[str(level)])
    return arr


def compute_intensities(darr: da.Array):
    """
    Compute robust intensity percentiles from a (t, c, z, y, x) tile array.
    Uses t=0, c=0 by default.
    """
    # take first timepoint + first channel
    sub = darr[0, 0]  # → (z, y, x)

    # load into memory
    vol = sub.compute()

    # handle NaNs
    v = vol[np.isfinite(vol)].ravel()

    p = np.percentile(v, [0, 0.1, 1, 50, 99, 99.9, 100])

    print("\n--- INTENSITY STATS ---")
    print(f"Shape: {vol.shape}")
    print(f"Min: {p[0]:.6g}")
    print(f"P0.1: {p[1]:.6g}")
    print(f"P1: {p[2]:.6g}")
    print(f"Median (P50): {p[3]:.6g}")
    print(f"P99: {p[4]:.6g}")
    print(f"P99.9: {p[5]:.6g}")
    print(f"Max: {p[6]:.6g}")
    print("\nSuggested robust window:", f"{p[2]:.4g} .. {p[5]:.4g}")

    return vol, p


def main():
    print(f"\nOpening tile group:\n  {BUCKET_TILE_PATH}")

    g = open_tile_group(BUCKET_TILE_PATH, anon=ANON)

    print(f"Loaded tile group. Keys: {list(g.keys())} (multiscale levels)")

    level = str(MULTISCALE_LEVEL)
    if level not in g:
        raise RuntimeError(f"Level {level} not found. Levels available: {list(g.keys())}")

    arr = load_multiscale_level(g, MULTISCALE_LEVEL)
    print(f"Loaded multiscale level {MULTISCALE_LEVEL}, Dask shape = {arr.shape}")

    compute_intensities(arr)


if __name__ == "__main__":
    main()
