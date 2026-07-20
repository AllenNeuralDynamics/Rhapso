import os
import csv
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Set, Tuple, Optional, Dict
from urllib.parse import urlparse
import boto3

# Controls how tall (vertically) the shift plots appear
INDEX_RANGE_MULTIPLIER = 5.0  # bump this up/down to taste


# ----------------------------
# Small helpers
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_affine_3x4(text: str) -> np.ndarray:
    vals = np.array(text.split(), dtype=float)
    if vals.size != 12:
        raise ValueError(f"Expected 12 values in 3x4 affine, got {vals.size}")
    return vals.reshape(3, 4)


def is_pure_translation(aff: np.ndarray, atol: float = 1e-9) -> bool:
    return np.allclose(aff[:, :3], np.eye(3), atol=atol)


def save_scatter(x, y, xlabel, ylabel, title, out_png):
    plt.figure()
    plt.scatter(x, y, s=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------
# Dropped-links helpers
# ----------------------------

def load_dropped_pairs(csv_path: str) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    Read a 'solver_removed_links' CSV and return:
      - dropped_pairs: set of undirected tile pairs (min(a,b), max(a,b))
      - pair_errors: dict mapping (min(a,b), max(a,b)) -> error value (float)

    Supports two formats:

    1) Old format (with type column):
        type,tp_a,a,tp_b,b,error
        solver_removed_link,0,7,0,10,32.8
        ...

       -> Only rows where type == "solver_removed_link" are used.

    2) New format (no type column):
        tp_a,a,tp_b,b,u,v,error
        0,7,0,10,7,10,32.8
        0,0,0,5,0,5,2.30
        ...

       -> Every row with valid a,b is treated as a dropped link.
    """
    dropped: Set[Tuple[int, int]] = set()
    pair_errors: Dict[Tuple[int, int], float] = {}

    try:
        with open(csv_path, "r", newline="") as f:
            # Try to auto-detect comma vs tab delimiter
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                reader = csv.DictReader(f, dialect=dialect)
            except csv.Error:
                # Fallback: standard comma-delimited
                reader = csv.DictReader(f)

            fieldnames = reader.fieldnames or []
            has_type = "type" in fieldnames

            for row in reader:
                # Skip obviously empty rows
                if not any(row.values()):
                    continue

                # Old style: filter specifically for solver_removed_link
                if has_type and row.get("type") != "solver_removed_link":
                    continue

                try:
                    a = int(row["a"])
                    b = int(row["b"])
                except (KeyError, ValueError, TypeError):
                    # If we can’t parse a/b, just skip that row
                    continue

                key = (min(a, b), max(a, b))
                dropped.add(key)

                # Error column (if present)
                err_val: Optional[float] = None
                err_str = row.get("error")
                if err_str is not None and err_str != "":
                    try:
                        err_val = float(err_str)
                    except ValueError:
                        err_val = None

                # If we have an error, keep the max (or first) per pair
                if err_val is not None:
                    if key in pair_errors:
                        pair_errors[key] = max(pair_errors[key], err_val)
                    else:
                        pair_errors[key] = err_val

        print(f"Loaded {len(dropped)} dropped pairs from {csv_path}: {sorted(dropped)}")
    except FileNotFoundError:
        print(f"[WARN] dropped_links file not found at {csv_path}; "
              "continuing without dropped-link labels.")
    except Exception as e:
        print(f"[WARN] Error reading dropped_links CSV at {csv_path}: {e!r}")

    return dropped, pair_errors


# ----------------------------
# Core logic
# ----------------------------

def get_nominal_grid(root: ET.Element):
    """
    From <ViewRegistrations>, read 'Translation to Nominal Grid' transforms
    and convert them into discrete (gridX, gridY, gridZ) indices.

    Returns:
        setup_to_grid: dict[setup_id] = (gridX, gridY, gridZ)
    """
    vr_root = root.find("ViewRegistrations")
    if vr_root is None:
        raise RuntimeError("No <ViewRegistrations> in XML")

    # setup -> nominal (x,y,z) translations
    setup_to_xyz = {}

    for vr in vr_root.findall("ViewRegistration"):
        setup = int(vr.get("setup"))
        nominal = None

        for vt in vr.findall("ViewTransform"):
            name = (vt.findtext("Name") or "").strip()
            if name != "Translation to Nominal Grid":
                continue

            aff_text = vt.findtext("affine")
            if not aff_text:
                continue

            aff = parse_affine_3x4(aff_text)
            if not is_pure_translation(aff):
                # nominal grid transforms should be pure translations
                continue

            nominal = aff[:, 3].astype(float)
            break

        if nominal is not None:
            setup_to_xyz[setup] = nominal

    if not setup_to_xyz:
        raise RuntimeError("No 'Translation to Nominal Grid' transforms found")

    # Convert continuous xyz to discrete grid indices
    xs = sorted({float(v[0]) for v in setup_to_xyz.values()})
    ys = sorted({float(v[1]) for v in setup_to_xyz.values()})
    zs = sorted({float(v[2]) for v in setup_to_xyz.values()})

    x_to_ix = {x: i for i, x in enumerate(xs)}
    y_to_iy = {y: i for i, y in enumerate(ys)}
    z_to_iz = {z: i for i, z in enumerate(zs)}

    setup_to_grid = {}
    for setup, xyz in setup_to_xyz.items():
        gx = x_to_ix[float(xyz[0])]
        gy = y_to_iy[float(xyz[1])]
        gz = z_to_iz[float(xyz[2])]
        setup_to_grid[setup] = (gx, gy, gz)

    return setup_to_grid


def extract_pairwise_rows(root: ET.Element, xy_thresh_log2: float):
    """
    Parse <StitchingResults> and return a list of rows:

    [TileA, TileB, ShiftX, ShiftY, ShiftZ,
     Correlation, OverlapX, OverlapY, OverlapZ, Alignment]

    Also deduplicates entries so that (a,b) with identical shift/corr isn't
    written twice.
    """
    sr = root.find("StitchingResults")
    if sr is None:
        raise RuntimeError("No <StitchingResults> in XML")

    rows = []
    seen = set()  # (min(a,b), max(a,b), sx, sy, sz, corr)

    for pr in sr.findall("PairwiseResult"):
        a = int(pr.get("view_setup_a"))
        b = int(pr.get("view_setup_b"))

        # 3x4 affine from <shift>
        shift_aff = parse_affine_3x4(pr.find("shift").text)
        if not is_pure_translation(shift_aff):
            raise SystemExit(f"Non-translation detected between {a} and {b}")

        shifts = shift_aff[:, 3].astype(float)  # (x,y,z)
        corr = float(pr.find("correlation").text)

        # Overlap bounding box -> extent in X/Y/Z
        bb = np.array(
            pr.find("overlap_boundingbox").text.split(),
            dtype=float
        ).reshape(2, 3)
        ext = bb[1] - bb[0]  # (dx, dy, dz)
        overlap_x, overlap_y, overlap_z = ext.tolist()

        # Classify orientation
        eps = 1e-9
        x = max(abs(overlap_x), eps)
        y = max(abs(overlap_y), eps)

        if np.log2(x / y) > xy_thresh_log2:
            align = "top_bottom"
        elif np.log2(y / x) > xy_thresh_log2:
            align = "left_right"
        else:
            align = "corner"

        sx_round = round(float(shifts[0]), 3)
        sy_round = round(float(shifts[1]), 3)
        sz_round = round(float(shifts[2]), 3)
        corr_round = round(corr, 6)

        key = (min(a, b), max(a, b), sx_round, sy_round, sz_round, corr_round)
        if key in seen:
            continue
        seen.add(key)

        rows.append([
            a, b,
            sx_round,
            sy_round,
            sz_round,
            corr_round,
            float(overlap_x),
            float(overlap_y),
            float(overlap_z),
            align,
        ])

    return rows


def write_pairwise_csv(
    rows_sorted,
    csv_path: str,
    dropped_pairs: Optional[Set[Tuple[int, int]]] = None,
):
    """
    Write pairwise links CSV locally.

    If dropped_pairs is provided, add a 'DroppedBySolver' yes/no column
    based on whether the (TileA, TileB) pair appears in the dropped set.
    """
    base_header = [
        "TileA", "TileB",
        "ShiftX", "ShiftY", "ShiftZ",
        "Correlation",
        "OverlapX", "OverlapY", "OverlapZ",
        "Alignment",
    ]
    if dropped_pairs is not None:
        header = base_header + ["DroppedBySolver"]
    else:
        header = base_header

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows_sorted:
            if dropped_pairs is None:
                w.writerow(r)
            else:
                a = int(r[0])
                b = int(r[1])
                key = (min(a, b), max(a, b))
                is_dropped = key in dropped_pairs
                w.writerow(list(r) + ["yes" if is_dropped else "no"])

def make_corr_and_shift_plots(
    rows_sorted,
    out_dir: str,
    dropped_pairs: Optional[Set[Tuple[int, int]]] = None,
):
    if not rows_sorted:
        return None, None, None

    corr = np.array([r[5] for r in rows_sorted], dtype=float)
    sx = np.array([r[2] for r in rows_sorted], dtype=float)
    sy = np.array([r[3] for r in rows_sorted], dtype=float)
    sz = np.array([r[4] for r in rows_sorted], dtype=float)
    rank = np.arange(1, len(rows_sorted) + 1)
    aligns = [r[9] for r in rows_sorted]

    dropped_flags = None
    if dropped_pairs is not None:
        dropped_flags = np.array(
            [
                (min(int(r[0]), int(r[1])), max(int(r[0]), int(r[1]))) in dropped_pairs
                for r in rows_sorted
            ],
            dtype=bool,
        )

    cat_order = ["top_bottom", "left_right", "corner"]
    cat_labels = {
        "top_bottom": "Top / Bottom",
        "left_right": "Left / Right",
        "corner": "Corner",
    }
    cat_colors = {
        "top_bottom": "tab:blue",
        "left_right": "tab:orange",
        "corner": "tab:green",
    }

    def axis_min_max(arr: np.ndarray):
        if arr.size == 0:
            return (-1.0, 1.0)

        vmin = float(arr.min())
        vmax = float(arr.max())

        if vmin == vmax:
            eps = 1.0 if vmin == 0 else abs(vmin) * 0.1
            return (vmin - eps, vmax + eps)

        pad = 0.05 * (vmax - vmin)
        return (vmin - pad, vmax + pad)

    # -------- Corr vs rank, split by category --------
    corr_png = os.path.join(out_dir, "corr_rank.png")

    fig, axes = plt.subplots(
        1,
        len(cat_order),
        figsize=(15, 4.5),
        sharey=True,
    )

    if len(cat_order) == 1:
        axes = [axes]

    for ax, cat in zip(axes, cat_order):
        idxs_cat = np.array(
            [i for i, a in enumerate(aligns) if a == cat],
            dtype=int,
        )

        ax.set_title(cat_labels[cat])
        ax.set_xlabel("Pair rank (corr desc)")
        ax.grid(True, alpha=0.25)

        if idxs_cat.size == 0:
            ax.text(
                0.5,
                0.5,
                "No pairs",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        if dropped_flags is None:
            ax.scatter(
                rank[idxs_cat],
                corr[idxs_cat],
                s=10,
                color=cat_colors[cat],
                label=cat_labels[cat],
            )
        else:
            local_dropped = dropped_flags[idxs_cat]
            kept_idx = idxs_cat[~local_dropped]
            drop_idx = idxs_cat[local_dropped]

            if kept_idx.size > 0:
                ax.scatter(
                    rank[kept_idx],
                    corr[kept_idx],
                    s=10,
                    color=cat_colors[cat],
                    label=cat_labels[cat],
                )

            if drop_idx.size > 0:
                ax.scatter(
                    rank[drop_idx],
                    corr[drop_idx],
                    s=40,
                    marker="x",
                    color=cat_colors[cat],
                )

        ax.text(
            0.02,
            0.96,
            f"n={idxs_cat.size}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "boxstyle": "round,pad=0.25",
                "alpha": 0.8,
            },
        )

    axes[0].set_ylabel("Correlation")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color=cat_colors["top_bottom"],
            label=cat_labels["top_bottom"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color=cat_colors["left_right"],
            label=cat_labels["left_right"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color=cat_colors["corner"],
            label=cat_labels["corner"],
        ),
    ]

    if dropped_flags is not None and dropped_flags.any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="None",
                color="black",
                label="Dropped by solver",
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.suptitle("Pairwise correlation ranked by orientation group", y=1.08)
    fig.tight_layout()
    fig.savefig(corr_png, dpi=200)
    plt.close(fig)

    # -------- Shifts plots, all links --------
    shift_arrays = [sx, sy, sz]
    axis_labels = ["X", "Y", "Z"]
    x_base = {cat: i for i, cat in enumerate(cat_order)}

    shifts_all_png = os.path.join(out_dir, "shifts_all.png")
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for ax, shifts, axis_label in zip(axes, shift_arrays, axis_labels):
        for cat in cat_order:
            idxs_cat = [i for i, a in enumerate(aligns) if a == cat]
            if not idxs_cat:
                continue

            idxs_cat = np.array(idxs_cat, dtype=int)
            y_vals = shifts[idxs_cat]

            n_cat = len(idxs_cat)
            if n_cat == 1:
                offsets = np.array([0.0])
            else:
                offsets = np.linspace(-0.4, 0.4, n_cat)

            x_vals = x_base[cat] + offsets

            if dropped_flags is None:
                ax.scatter(
                    x_vals,
                    y_vals,
                    s=10,
                    color=cat_colors[cat],
                    label=cat_labels[cat],
                )
            else:
                local_dropped = dropped_flags[idxs_cat]
                keep_idx = np.where(~local_dropped)[0]
                drop_idx = np.where(local_dropped)[0]

                if keep_idx.size > 0:
                    ax.scatter(
                        x_vals[keep_idx],
                        y_vals[keep_idx],
                        s=10,
                        color=cat_colors[cat],
                        label=cat_labels[cat],
                    )

                if drop_idx.size > 0:
                    ax.scatter(
                        x_vals[drop_idx],
                        y_vals[drop_idx],
                        s=30,
                        color="red",
                        marker="x",
                    )

        ymin, ymax = axis_min_max(shifts)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(f"Shift{axis_label} (pixels)")
        ax.set_title(f"Shift{axis_label} by orientation group")

    axes[-1].set_xticks(list(x_base.values()))
    axes[-1].set_xticklabels(
        [cat_labels[c] for c in cat_order],
        rotation=15,
    )
    axes[-1].set_xlabel("Link orientation group")

    fig.tight_layout()
    fig.savefig(shifts_all_png, dpi=200)
    plt.close(fig)

    # -------- Shifts kept only, one chart per orientation category --------
    if dropped_flags is not None and dropped_flags.any():
        kept_flags = ~dropped_flags
        shifts_kept_png = None

        if kept_flags.any():
            for cat in cat_order:
                kept_cat_idxs = [
                    i
                    for i in range(len(aligns))
                    if kept_flags[i] and aligns[i] == cat
                ]

                if not kept_cat_idxs:
                    print(f"No kept pairs for {cat_labels[cat]}")
                    continue

                kept_cat_idxs = np.array(kept_cat_idxs, dtype=int)

                kept_cat_png = os.path.join(out_dir, f"shifts_kept_{cat}.png")

                fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

                for ax, shifts, axis_label in zip(axes, shift_arrays, axis_labels):
                    y_vals = shifts[kept_cat_idxs]

                    n_cat = len(kept_cat_idxs)
                    if n_cat == 1:
                        x_vals = np.array([0.0])
                    else:
                        x_vals = np.linspace(-0.4, 0.4, n_cat)

                    ax.scatter(
                        x_vals,
                        y_vals,
                        s=10,
                        color=cat_colors[cat],
                        label=cat_labels[cat],
                    )

                    ymin, ymax = axis_min_max(y_vals)
                    ax.set_ylim(ymin, ymax)
                    ax.set_ylabel(f"Shift{axis_label} (pixels)")
                    ax.set_title(f"Shift{axis_label} for {cat_labels[cat]} kept only")
                    ax.grid(True, alpha=0.25)

                axes[-1].set_xticks([0.0])
                axes[-1].set_xticklabels([cat_labels[cat]], rotation=15)
                axes[-1].set_xlabel("Link orientation group")

                fig.suptitle(
                    f"Pairwise shifts for {cat_labels[cat]} kept only",
                    y=1.02,
                )

                fig.tight_layout()
                fig.savefig(kept_cat_png, dpi=200)
                plt.close(fig)

                print(f"Kept shift plot for {cat_labels[cat]}:", kept_cat_png)

                if shifts_kept_png is None:
                    shifts_kept_png = kept_cat_png

            if shifts_kept_png is None:
                shifts_kept_png = shifts_all_png
        else:
            shifts_kept_png = shifts_all_png
    else:
        shifts_kept_png = shifts_all_png

    print("Pair counts by orientation:")
    for cat in cat_order:
        n_cat = sum(1 for a in aligns if a == cat)
        print(f"  {cat_labels[cat]}: {n_cat}")

    if dropped_flags is not None:
        print("Dropped pair count:", int(dropped_flags.sum()))

    print("ShiftX min/max:", float(sx.min()), float(sx.max()))
    print("ShiftY min/max:", float(sy.min()), float(sy.max()))
    print("ShiftZ min/max:", float(sz.min()), float(sz.max()))

    return corr_png, shifts_all_png, shifts_kept_png

def write_dropped_links_report(
    rows_sorted,
    dropped_pairs: Optional[Set[Tuple[int, int]]],
    pair_errors: Dict[Tuple[int, int], float],
    out_txt_path: str,
) -> None:
    """
    Write a plain-text report listing all dropped links that we can
    match to pairwise metrics from the XML.

    For each dropped (TileA, TileB) pair, prints:
      - TileA, TileB
      - Error (from dropped-links CSV)
      - Corr (single corr value derived from XML; we use max corr per pair)
    """
    if not dropped_pairs:
        return

    # Map pair -> list of corr values from XML
    corr_by_pair: Dict[Tuple[int, int], list] = {}
    for r in rows_sorted:
        a = int(r[0])
        b = int(r[1])
        corr = float(r[5])
        key = (min(a, b), max(a, b))
        if key in dropped_pairs:
            corr_by_pair.setdefault(key, []).append(corr)

    with open(out_txt_path, "w") as f:
        f.write("Dropped links summary\n")
        f.write("=====================\n\n")
        f.write(f"Total dropped pairs (from CSV): {len(dropped_pairs)}\n")
        f.write(f"Dropped pairs found in XML pairwise results: {len(corr_by_pair)}\n\n")

        # Any dropped pairs that didn't show up in StitchingResults
        missing_pairs = sorted(dropped_pairs - set(corr_by_pair.keys()))
        if missing_pairs:
            f.write("Dropped pairs NOT found in StitchingResults (by TileA/TileB):\n")
            for (a, b) in missing_pairs:
                err = pair_errors.get((a, b))
                if err is not None:
                    f.write(f"  - ({a}, {b})  Error={err}\n")
                else:
                    f.write(f"  - ({a}, {b})  Error=N/A\n")
            f.write("\n")

        # Summary for each dropped pair we did find
        if not corr_by_pair:
            f.write("No dropped pairs had corresponding pairwise correlations in the XML.\n")
            return

        f.write("Dropped pairs with error + corr:\n")
        for key in sorted(corr_by_pair.keys()):
            a, b = key
            best_corr = max(corr_by_pair[key])
            err = pair_errors.get(key)
            if err is None:
                err_str = "N/A"
            else:
                err_str = f"{err}"
            f.write(f"Pair (TileA={a}, TileB={b}):  Error={err_str},  Corr={best_corr}\n")


def run_qc(
    xml_path: str,
    out_dir: str,
    xy_thresh_log2: float,
    bad_corr_thresh: float,
    dropped_csv_path: Optional[str] = None,
) -> None:
    """
    High-level driver: do everything for one XML.

    If dropped_csv_path is provided and exists, use it to label links
    dropped by the iterative solver.
    """
    ensure_dir(out_dir)

    if xml_path.startswith("s3://"):
        parsed = urlparse(xml_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        xml_bytes = obj["Body"].read()

        root = ET.fromstring(xml_bytes)
    else:
        root = ET.parse(xml_path).getroot()

    # grid layout (for mapping bad links)
    setup_to_grid = get_nominal_grid(root)

    # pairwise data (deduped)
    rows = extract_pairwise_rows(root, xy_thresh_log2)
    rows_sorted = sorted(rows, key=lambda r: r[5], reverse=True)  # by corr desc

    # Dropped pairs (if CSV given)
    dropped_pairs: Optional[Set[Tuple[int, int]]] = None
    pair_errors: Dict[Tuple[int, int], float] = {}
    if dropped_csv_path is not None:
        dropped_pairs_loaded, pair_errors_loaded = load_dropped_pairs(dropped_csv_path)
        if dropped_pairs_loaded:
            dropped_pairs = dropped_pairs_loaded
            pair_errors = pair_errors_loaded

    # CSV
    csv_path = os.path.join(out_dir, "pairwise_links.csv")
    write_pairwise_csv(rows_sorted, csv_path, dropped_pairs)

    # Text report of dropped links (pair + error + corr)
    if dropped_pairs:
        dropped_txt_path = os.path.join(out_dir, "dropped_links_metrics.txt")
        write_dropped_links_report(rows_sorted, dropped_pairs, pair_errors, dropped_txt_path)
        print("  Dropped links report:", dropped_txt_path)

    # plots
    make_corr_and_shift_plots(rows_sorted, out_dir, dropped_pairs)

    print("QC done.")
    print("  CSV :", csv_path)
    print("  PNGs:", out_dir)

if __name__ == "__main__":
    XML_PATH = "s3://aind-open-data/HCR_831988-s1-ls2_2026-05-27_00-00-00_processed_2026-05-28_01-30-18/image_tile_alignment/bigstitcher.xml"
    DROPPED_CSV_PATH = "s3://aind-open-data/HCR_831988-s1-ls2_2026-05-27_00-00-00_processed_2026-05-28_01-30-18/image_tile_alignment/solver_removed_links.csv"
    OUT_DIR  = "/Users/sean.fite/Desktop/pairwise_qc_out"

    XY_THRESH_LOG2  = 2.0   # same meaning as your original xy_thresh
    BAD_CORR_THRESH = 0.99  # links below this are drawn as 'bad' on the grid

    run_qc(XML_PATH, OUT_DIR, XY_THRESH_LOG2, BAD_CORR_THRESH, DROPPED_CSV_PATH)