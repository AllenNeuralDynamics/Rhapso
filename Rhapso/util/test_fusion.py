import csv
import json
import math
from urllib.parse import urlparse

import boto3

TSV_PATH = "/Users/sean.fite/Desktop/fusion_blocks_python.tsv"
ZARR_S3  = "s3://aind-scratch-data/sean.fite/new_affine_fusion/test_150/fusion/fused.zarr"
ARRAY_PATH = "0"

# For your OME-Zarr fused output, assume [t,c,z,y,x] and we check t=0,c=0
T_INDEX = 0
C_INDEX = 0

# If True, dedupe expected chunk indices (recommended)
DEDUP_EXPECTED = True

# Safety cap for quick tests (None = all rows)
MAX_TSV_ROWS = None

OUT_MISSING = "missing_chunks_from_list.tsv"


def parse_s3_uri(s3_uri: str):
    u = urlparse(s3_uri)
    if u.scheme != "s3":
        raise ValueError(f"Not an s3 uri: {s3_uri}")
    return u.netloc, u.path.lstrip("/")


def s3_read_json(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def cell_to_chunk_ranges_xyz(x0, x1, y0, y1, z0, z1, cz, cy, cx):
    # TSV highs are inclusive -> make exclusive
    x1e, y1e, z1e = x1 + 1, y1 + 1, z1 + 1

    x0c = x0 // cx
    x1c = (x1e - 1) // cx
    y0c = y0 // cy
    y1c = (y1e - 1) // cy
    z0c = z0 // cz
    z1c = (z1e - 1) // cz
    return x0c, x1c, y0c, y1c, z0c, z1c


def list_existing_zyx(s3, bucket, prefix_key):
    """
    List all objects under prefix_key and return a set of (z,y,x) tuples.
    prefix_key should be: <base>/0/<t>/<c>/  (ends with slash)
    """
    paginator = s3.get_paginator("list_objects_v2")
    existing = set()
    n_objs = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_key):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # key looks like: .../0/0/0/z/y/x
            rel = key[len(prefix_key):].strip("/")
            if not rel:
                continue
            parts = rel.split("/")
            if len(parts) != 3:
                # ignore metadata files if any
                continue
            try:
                z = int(parts[0]); y = int(parts[1]); x = int(parts[2])
            except ValueError:
                continue
            existing.add((z, y, x))
            n_objs += 1

    return existing, n_objs


def main():
    s3 = boto3.client("s3")

    bucket, base_key = parse_s3_uri(ZARR_S3)
    base_key = base_key.rstrip("/")

    # read zarr metadata for chunks
    zarray_key = f"{base_key}/{ARRAY_PATH}/.zarray"
    meta = s3_read_json(s3, bucket, zarray_key)

    shape = meta["shape"]
    chunks = meta["chunks"]
    print("Zarr meta:")
    print(f"  .zarray key : s3://{bucket}/{zarray_key}")
    print(f"  shape       : {shape}")
    print(f"  chunks      : {chunks}")
    print()

    if len(chunks) < 3:
        raise RuntimeError(f"Expected >=3D, got chunks={chunks}")

    # spatial chunk sizes are last 3 (z,y,x)
    cz, cy, cx = chunks[-3], chunks[-2], chunks[-1]
    print(f"Using spatial chunks (cz,cy,cx)=({cz},{cy},{cx})")

    # 1) LIST existing chunk objects under /0/t/c/
    prefix_key = f"{base_key}/{ARRAY_PATH}/{T_INDEX}/{C_INDEX}/"
    print(f"\nListing existing chunks under: s3://{bucket}/{prefix_key}")

    existing, n_objs = list_existing_zyx(s3, bucket, prefix_key)
    print(f"  listed objects parsed as chunks: {len(existing)} (raw objects seen={n_objs})")

    # 2) Compute expected chunk indices from TSV
    expected = set() if DEDUP_EXPECTED else []
    total_rows = 0

    with open(TSV_PATH, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            total_rows += 1
            if MAX_TSV_ROWS is not None and total_rows > MAX_TSV_ROWS:
                break

            x0 = int(row["x0"]); x1 = int(row["x1"])
            y0 = int(row["y0"]); y1 = int(row["y1"])
            z0 = int(row["z0"]); z1 = int(row["z1"])

            x0c, x1c, y0c, y1c, z0c, z1c = cell_to_chunk_ranges_xyz(
                x0, x1, y0, y1, z0, z1, cz, cy, cx
            )

            for zc in range(z0c, z1c + 1):
                for yc in range(y0c, y1c + 1):
                    for xc in range(x0c, x1c + 1):
                        if DEDUP_EXPECTED:
                            expected.add((zc, yc, xc))
                        else:
                            expected.append((zc, yc, xc))

            if total_rows % 20000 == 0:
                print(f"  [progress] TSV rows={total_rows} expected_unique={len(expected) if DEDUP_EXPECTED else 'n/a'}")

    if not DEDUP_EXPECTED:
        expected = set(expected)

    print(f"\nTSV rows read: {total_rows}")
    print(f"Expected unique chunks from TSV: {len(expected)}")

    # 3) Diff
    missing = sorted(expected - existing)
    extra = sorted(existing - expected)

    print(f"\nMissing chunks (expected-but-not-listed): {len(missing)}")
    print(f"Extra chunks (listed-but-not-expected):   {len(extra)}")

    # 4) Write missing
    with open(OUT_MISSING, "w", newline="") as wf:
        w = csv.writer(wf, delimiter="\t")
        w.writerow(["z_chunk", "y_chunk", "x_chunk", "s3_key"])
        for (z, y, x) in missing:
            w.writerow([z, y, x, f"s3://{bucket}/{prefix_key}{z}/{y}/{x}"])

    print(f"\nWrote: {OUT_MISSING}")

    # Print a few examples
    if missing:
        print("\nFirst 20 missing:")
        for z, y, x in missing[:20]:
            print(f"  {z}/{y}/{x}")

    if extra:
        print("\nFirst 20 extra:")
        for z, y, x in extra[:20]:
            print(f"  {z}/{y}/{x}")


if __name__ == "__main__":
    main()