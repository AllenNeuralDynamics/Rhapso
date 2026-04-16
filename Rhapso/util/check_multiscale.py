#!/usr/bin/env python3
import numpy as np
import s3fs
import zarr

S3_ZARR = "s3://aind-open-data/HCR_000000-s107-ls1_2026-01-23_00-00-00_processed_2026-01-24_06-00-53/image_tile_fusing/fused/channel_488.zarr"

def human(n):
    for u in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if n < 1024 or u == "PiB":
            return f"{n:.2f} {u}" if u != "B" else f"{int(n)} B"
        n /= 1024

fs = s3fs.S3FileSystem(anon=True)

root = S3_ZARR.replace("s3://", "", 1).rstrip("/")  # bucket/key...
store = s3fs.S3Map(root=root, s3=fs, check=False)
g = zarr.open_group(store=store, mode="r")

print("Zarr:", S3_ZARR)
print()

for level in range(6):  # 0..6
    p = str(level)
    a = zarr.open_array(store=store, path=p, mode="r")  # each level folder is an array
    logical = int(np.prod(a.shape) * a.dtype.itemsize)   # uncompressed in-memory size
    stored = fs.du(f"{root}/{p}", total=True)            # bytes actually stored in S3 under that level

    print(
        f"Level {p}: shape={a.shape}  chunks={a.chunks}  dtype={a.dtype}  "
        f"logical={human(logical)}  stored={human(stored)}"
    )