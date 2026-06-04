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
    """
    Plots:
      - corr vs rank (all) -> 01_corr_rank.png
      - three stacked subplots for ShiftX / ShiftY / ShiftZ with
        orientation groups along X (Top/Bottom, Left/Right, Corner),
        and points spread horizontally within each group -> 02_shifts_all.png

      If dropped_pairs is provided, dropped links are highlighted.
    """
    # ----- pull arrays out of rows -----
    corr = np.array([r[5] for r in rows_sorted], dtype=float)
    sx   = np.array([r[2] for r in rows_sorted], dtype=float)
    sy   = np.array([r[3] for r in rows_sorted], dtype=float)
    sz   = np.array([r[4] for r in rows_sorted], dtype=float)
    rank = np.arange(1, len(rows_sorted) + 1)      # 1..N for corr plot
    aligns = [r[9] for r in rows_sorted]           # "top_bottom", "left_right", "corner"

    # ----- dropped flags -----
    dropped_flags = None
    if dropped_pairs is not None:
        dropped_flags = np.array(
            [
                (min(int(r[0]), int(r[1])), max(int(r[0]), int(r[1]))) in dropped_pairs
                for r in rows_sorted
            ],
            dtype=bool,
        )

    # ----- correlation vs rank -----
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if dropped_flags is None or not dropped_flags.any():
        ax.scatter(rank, corr, s=10)
    else:
        kept = ~dropped_flags
        ax.scatter(rank[kept], corr[kept], s=10, label="Kept")
        ax.scatter(
            rank[dropped_flags],
            corr[dropped_flags],
            s=30,
            color="red",
            marker="x",
            label="Dropped by solver",
        )
        ax.legend()

    ax.set_xlabel("Pair rank (corr desc)")
    ax.set_ylabel("Correlation")
    ax.set_title("Pairwise correlation (ranked)")
    fig.tight_layout()
    out_corr = os.path.join(out_dir, "01_corr_rank.png")
    fig.savefig(out_corr, dpi=200)
    plt.close(fig)

    # ----- helper for min/max with a tiny pad -----
    def axis_min_max(arr: np.ndarray):
        """Return (ymin, ymax) that span full data with a small padding."""
        if arr.size == 0:
            return (-1.0, 1.0)
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmin == vmax:
            eps = 1.0 if vmin == 0 else abs(vmin) * 0.1
            return (vmin - eps, vmax + eps)
        pad = 0.05 * (vmax - vmin)
        return (vmin - pad, vmax + pad)

    # ----- grouped / jittered shift plots -----
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    shift_arrays = [sx, sy, sz]
    axis_labels  = ["X", "Y", "Z"]

    cat_order = ["top_bottom", "left_right", "corner"]
    cat_labels = {
        "top_bottom": "Top / Bottom",
        "left_right": "Left / Right",
        "corner":     "Corner",
    }
    # base x positions for each category group
    x_base = {cat: i for i, cat in enumerate(cat_order)}

    for ax, shifts, axis_label in zip(axes, shift_arrays, axis_labels):
        for cat in cat_order:
            idxs_cat = [i for i, a in enumerate(aligns) if a == cat]
            if not idxs_cat:
                continue

            idxs_cat = np.array(idxs_cat, dtype=int)
            y_vals = shifts[idxs_cat]

            # Spread points horizontally within the group using an index-based offset
            n_cat = len(idxs_cat)
            if n_cat == 1:
                offsets = np.array([0.0])
            else:
                # Evenly space in [-0.4, 0.4]
                offsets = np.linspace(-0.4, 0.4, n_cat)

            x_vals = x_base[cat] + offsets

            if dropped_flags is None:
                ax.scatter(x_vals, y_vals, s=10, label=cat_labels[cat])
            else:
                local_dropped = dropped_flags[idxs_cat]
                keep_idx = np.where(~local_dropped)[0]
                drop_idx = np.where(local_dropped)[0]

                if keep_idx.size > 0:
                    ax.scatter(
                        x_vals[keep_idx],
                        y_vals[keep_idx],
                        s=10,
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

        # Y limits = full min/max of this shift axis (with small pad)
        ymin, ymax = axis_min_max(shifts)
        ax.set_ylim(ymin, ymax)

        ax.set_ylabel(f"Shift{axis_label} (pixels)")
        ax.set_title(f"Shift{axis_label} by orientation group")

    # Shared X axis: group labels at integer positions
    axes[-1].set_xticks(list(x_base.values()))
    axes[-1].set_xticklabels([cat_labels[c] for c in cat_order], rotation=15)
    axes[-1].set_xlabel("Link orientation group")

    plt.tight_layout()
    out_png = os.path.join(out_dir, "02_shifts_all.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    # Optional console stats
    print("ShiftX min/max:", float(sx.min()))
    print("ShiftY min/max:", float(sy.min()))
    print("ShiftZ min/max:", float(sz.min()))

def make_bad_links_grid_plot(
    rows,
    setup_to_grid,
    bad_corr_thresh: float,
    out_dir: str,
    dropped_pairs: Optional[Set[Tuple[int, int]]] = None,
):
    """
    Draw ONE stretched grid map of all pairwise links.

    - Each tile is drawn as a square marker
    - All pairwise links are drawn on one grid
    - Link color is based on correlation band
    - For skinny layouts like 2 cols x many rows, X is stretched for display
    - Legend/title layout is tightened so there is less empty space above the grid
    """
    if not setup_to_grid:
        return

    all_gx = [g[0] for g in setup_to_grid.values()]
    all_gy = [g[1] for g in setup_to_grid.values()]

    min_gx = min(all_gx)
    max_gx = max(all_gx)
    min_gy = min(all_gy)
    max_gy = max(all_gy)

    grid_w = max_gx - min_gx + 1
    grid_h = max_gy - min_gy + 1

    print("[grid-debug] x index min/max:", min_gx, max_gx, "unique:", sorted(set(all_gx)))
    print("[grid-debug] y index min/max:", min_gy, max_gy, "unique:", sorted(set(all_gy)))
    print("[grid-debug] grid_w/grid_h:", grid_w, grid_h)

    # Display-only X stretch for skinny maps.
    # Reduced from the earlier wider stretch so tiles look less wide / more square.
    if grid_w <= 3 and grid_h >= 8:
        x_stretch = 2.75
    elif grid_w <= 5 and grid_h >= 12:
        x_stretch = 2.0
    else:
        x_stretch = 1.0

    print("[grid-debug] display x_stretch:", x_stretch)

    def x_plot(gx: int) -> float:
        return (gx - min_gx) * x_stretch + min_gx

    def edge_color(corr: float) -> str:
        if corr >= 0.90:
            return "tab:blue"
        elif corr >= 0.80:
            return "tab:green"
        elif corr >= 0.70:
            return "gold"
        else:
            return "red"

    # Build pair -> best row by max corr.
    best_row_by_pair: Dict[Tuple[int, int], list] = {}
    for r in rows:
        a = int(r[0])
        b = int(r[1])
        corr = float(r[5])
        key = (min(a, b), max(a, b))

        if key not in best_row_by_pair or corr > float(best_row_by_pair[key][5]):
            best_row_by_pair[key] = r

    dropped_rows = []
    dropped_missing_from_xml = []
    dropped_missing_from_grid = []

    if dropped_pairs:
        for key in sorted(dropped_pairs):
            r = best_row_by_pair.get(key)
            if r is None:
                dropped_missing_from_xml.append(key)
                continue

            a = int(r[0])
            b = int(r[1])

            if a not in setup_to_grid or b not in setup_to_grid:
                dropped_missing_from_grid.append(key)
                continue

            dropped_rows.append(r)

    # Figure sizing.
    # Slightly narrower and a bit taller so the map feels more square overall.
    display_w = (grid_w - 1) * x_stretch + 1
    fig_w = max(7.0, min(14.0, display_w * 1.6))
    fig_h = max(14.0, min(38.0, grid_h * 0.50))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    tile_fontsize = 8 if grid_h <= 30 else 6 if grid_h <= 60 else 5
    tile_marker_size = 180 if grid_h <= 30 else 120 if grid_h <= 60 else 80

    def draw_nodes():
        for setup, (gx, gy, gz) in setup_to_grid.items():
            y_plot = max_gy - gy
            px = x_plot(gx)

            ax.scatter(
                px,
                y_plot,
                s=tile_marker_size,
                color="white",
                edgecolors="black",
                marker="s",
                linewidths=1.0,
                zorder=5,
            )

            ax.text(
                px,
                y_plot,
                str(setup),
                fontsize=tile_fontsize,
                color="black",
                ha="center",
                va="center",
                zorder=6,
            )

    def draw_edge(
        a: int,
        b: int,
        corr: float,
        *,
        linestyle: str,
        linewidth: float,
        zorder: int,
        color_override=None,
        alpha: float = 0.9,
    ):
        if a not in setup_to_grid or b not in setup_to_grid:
            return False

        gx_a, gy_a, _ = setup_to_grid[a]
        gx_b, gy_b, _ = setup_to_grid[b]

        ay = max_gy - gy_a
        by = max_gy - gy_b

        ax.plot(
            [x_plot(gx_a), x_plot(gx_b)],
            [ay, by],
            linewidth=linewidth,
            linestyle=linestyle,
            color=edge_color(corr) if color_override is None else color_override,
            alpha=alpha,
            zorder=zorder,
        )
        return True

    drawn_all = 0
    skipped_all = 0

    for r in rows:
        a = int(r[0])
        b = int(r[1])
        corr = float(r[5])

        ok = draw_edge(
            a,
            b,
            corr,
            linestyle="-",
            linewidth=1.2,
            zorder=2,
        )

        if ok:
            drawn_all += 1
        else:
            skipped_all += 1

    drawn_dropped = 0

    if dropped_rows:
        for r in dropped_rows:
            a = int(r[0])
            b = int(r[1])
            corr = float(r[5])

            ok = draw_edge(
                a,
                b,
                corr,
                linestyle=":",
                linewidth=2.5,
                zorder=4,
                color_override="black",
                alpha=1.0,
            )

            if ok:
                drawn_dropped += 1

    # Draw tiles last so squares/labels stay above links.
    draw_nodes()

    legend_handles = [
        Line2D([0], [0], color="tab:blue", lw=2, label="corr ≥ 0.90"),
        Line2D([0], [0], color="tab:green", lw=2, label="0.80 ≤ corr < 0.90"),
        Line2D([0], [0], color="gold", lw=2, label="0.70 ≤ corr < 0.80"),
        Line2D([0], [0], color="red", lw=2, label="corr < 0.70"),
    ]

    if dropped_pairs:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle=":",
                label="Dropped by solver",
            )
        )

    # Pull legend closer to the title/grid.
    fig.legend(
        handles=legend_handles,
        title="Link bands",
        loc="upper center",
        ncol=1,
        bbox_to_anchor=(0.5, 0.945),
        borderaxespad=0.1,
        frameon=True,
    )

    ax.set_title(
        "All pairwise links on grid (colored by corr band)",
        fontsize=12,
        pad=4,
    )
    ax.set_xlabel("Grid X index")
    ax.set_ylabel("Grid Y index")
    ax.set_aspect("equal", adjustable="box")

    pad = 0.75
    ax.set_xlim(x_plot(min_gx) - pad, x_plot(max_gx) + pad)
    ax.set_ylim(-pad, max_gy + pad)

    x_ticks = [x_plot(gx) for gx in range(min_gx, max_gx + 1)]
    x_labels = [str(gx) for gx in range(min_gx, max_gx + 1)]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(0, max_gy + 1))

    ax.tick_params(axis="x", labelsize=9, rotation=0)
    ax.tick_params(axis="y", labelsize=8)

    ax.grid(True, alpha=0.25)

    # Reserve less top space so the gap is much smaller.
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.905])

    out_png = os.path.join(out_dir, "05_all_pairwise_links_on_grid.png")
    fig.savefig(out_png, dpi=250, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)

    print("[grid-debug] total pairwise rows:", len(rows))
    print("[grid-debug] drawn all links:", drawn_all)
    print("[grid-debug] skipped all links:", skipped_all)

    if dropped_pairs:
        print("[grid-debug] dropped pairs from CSV:", len(dropped_pairs))
        print("[grid-debug] drawn dropped links:", drawn_dropped)
        print("[grid-debug] dropped missing from XML pairwise rows:", len(dropped_missing_from_xml))
        print("[grid-debug] dropped missing grid coords:", len(dropped_missing_from_grid))

        if dropped_missing_from_xml:
            print("[grid-debug] first dropped missing from XML:", dropped_missing_from_xml[:25])
        if dropped_missing_from_grid:
            print("[grid-debug] first dropped missing grid coords:", dropped_missing_from_grid[:25])

    print("[grid-debug] wrote grid map:", out_png)

# def make_bad_links_grid_plot(
#     rows,
#     setup_to_grid,
#     bad_corr_thresh: float,
#     out_dir: str,
#     dropped_pairs: Optional[Set[Tuple[int, int]]] = None,
# ):
#     """
#     Draw tiles at their nominal grid indices.

#     Solid lines:
#       - pairwise links with corr < bad_corr_thresh

#     Dotted lines:
#       - all solver-dropped links found in XML pairwise rows,
#         even if corr >= bad_corr_thresh

#     Notes:
#       - Explicit axis limits/ticks prevent edge rows/columns from getting clipped.
#       - Figure size only changes rendering size; x/y limits determine visible grid range.
#     """
#     if not setup_to_grid:
#         return

#     all_gx = [g[0] for g in setup_to_grid.values()]
#     all_gy = [g[1] for g in setup_to_grid.values()]

#     min_gx = min(all_gx)
#     max_gx = max(all_gx)
#     min_gy = min(all_gy)
#     max_gy = max(all_gy)

#     print("[grid-debug] x index min/max:", min_gx, max_gx, "unique:", sorted(set(all_gx)))
#     print("[grid-debug] y index min/max:", min_gy, max_gy, "unique:", sorted(set(all_gy)))

#     fig, ax = plt.subplots(figsize=(20, 20))

#     # Draw tile nodes
#     for setup, (gx, gy, gz) in setup_to_grid.items():
#         y_plot = max_gy - gy
#         ax.scatter(gx, y_plot, s=70, color="black", zorder=5)
#         ax.text(
#             gx + 0.05,
#             y_plot + 0.05,
#             str(setup),
#             fontsize=7,
#             color="black",
#             zorder=6,
#         )

#     def edge_color(corr: float) -> str:
#         if corr >= 0.90:
#             return "tab:blue"
#         elif corr >= 0.80:
#             return "tab:green"
#         elif corr >= 0.70:
#             return "gold"
#         else:
#             return "red"

#     def draw_edge(a: int, b: int, corr: float, *, linestyle: str, linewidth: float, zorder: int):
#         if a not in setup_to_grid or b not in setup_to_grid:
#             return False

#         gx_a, gy_a, _ = setup_to_grid[a]
#         gx_b, gy_b, _ = setup_to_grid[b]

#         ay = max_gy - gy_a
#         by = max_gy - gy_b

#         ax.plot(
#             [gx_a, gx_b],
#             [ay, by],
#             linewidth=linewidth,
#             linestyle=linestyle,
#             color=edge_color(corr),
#             alpha=0.9,
#             zorder=zorder,
#         )
#         return True

#     # Build pair -> best row by max corr.
#     # Used so dropped links can be drawn even if corr >= bad_corr_thresh.
#     best_row_by_pair: Dict[Tuple[int, int], list] = {}
#     for r in rows:
#         a = int(r[0])
#         b = int(r[1])
#         corr = float(r[5])
#         key = (min(a, b), max(a, b))

#         if key not in best_row_by_pair or corr > float(best_row_by_pair[key][5]):
#             best_row_by_pair[key] = r

#     # Draw solid bad-correlation links
#     bad_rows = [r for r in rows if float(r[5]) < bad_corr_thresh]

#     drawn_bad = 0
#     skipped_bad = 0

#     for r in bad_rows:
#         a = int(r[0])
#         b = int(r[1])
#         corr = float(r[5])

#         if draw_edge(a, b, corr, linestyle="-", linewidth=1.0, zorder=2):
#             drawn_bad += 1
#         else:
#             skipped_bad += 1

#     # Draw all dropped links as dotted, not only bad_rows
#     drawn_dropped = 0
#     dropped_missing_from_xml = []
#     dropped_missing_from_grid = []

#     if dropped_pairs:
#         for key in sorted(dropped_pairs):
#             r = best_row_by_pair.get(key)
#             if r is None:
#                 dropped_missing_from_xml.append(key)
#                 continue

#             a = int(r[0])
#             b = int(r[1])
#             corr = float(r[5])

#             if draw_edge(a, b, corr, linestyle=":", linewidth=2.5, zorder=4):
#                 drawn_dropped += 1
#             else:
#                 dropped_missing_from_grid.append(key)

#     print("[grid-debug] total pairwise rows:", len(rows))
#     print("[grid-debug] bad rows corr<thresh:", len(bad_rows))
#     print("[grid-debug] drawn bad links:", drawn_bad)
#     print("[grid-debug] skipped bad links missing grid coords:", skipped_bad)

#     if dropped_pairs:
#         print("[grid-debug] dropped pairs from CSV:", len(dropped_pairs))
#         print("[grid-debug] drawn dropped links:", drawn_dropped)
#         print("[grid-debug] dropped missing from XML pairwise rows:", len(dropped_missing_from_xml))
#         print("[grid-debug] dropped missing grid coords:", len(dropped_missing_from_grid))

#         if dropped_missing_from_xml:
#             print("[grid-debug] first dropped missing from XML:", dropped_missing_from_xml[:25])
#         if dropped_missing_from_grid:
#             print("[grid-debug] first dropped missing grid coords:", dropped_missing_from_grid[:25])

#     legend_handles = [
#         Line2D([0], [0], color="tab:blue", lw=2, label="corr ≥ 0.90"),
#         Line2D([0], [0], color="tab:green", lw=2, label="0.80 ≤ corr < 0.90"),
#         Line2D([0], [0], color="gold", lw=2, label="0.70 ≤ corr < 0.80"),
#         Line2D([0], [0], color="red", lw=2, label="corr < 0.70"),
#     ]

#     if dropped_pairs:
#         legend_handles.append(
#             Line2D(
#                 [0],
#                 [0],
#                 color="black",
#                 lw=2,
#                 linestyle=":",
#                 label="Dropped by solver",
#             )
#         )

#     fig.legend(
#         handles=legend_handles,
#         title="Link bands",
#         loc="upper center",
#         ncol=1,
#         bbox_to_anchor=(0.5, 0.97),
#         borderaxespad=0.2,
#     )

#     ax.set_title(
#         f"Low-corr links + solver-dropped links "
#         f"(solid corr < {bad_corr_thresh}, dotted dropped)"
#     )
#     ax.set_xlabel("Grid X index")
#     ax.set_ylabel("Grid Y index")
#     ax.set_aspect("equal", adjustable="box")

#     # Critical: explicitly show every grid index, including 11+
#     pad = 0.75
#     ax.set_xlim(min_gx - pad, max_gx + pad)
#     ax.set_ylim(-pad, max_gy + pad)

#     ax.set_xticks(range(min_gx, max_gx + 1))
#     ax.set_yticks(range(0, max_gy + 1))

#     ax.grid(True, alpha=0.25)

#     fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.84])

#     out_png = os.path.join(out_dir, "05_bad_links_grid.png")
#     fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.25)
#     plt.close(fig)


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
    make_bad_links_grid_plot(rows, setup_to_grid, bad_corr_thresh, out_dir, dropped_pairs)

    print("QC done.")
    print("  CSV :", csv_path)
    print("  PNGs:", out_dir)


# ----------------------------
# Tiny main: just hard-coded params
# ----------------------------

if __name__ == "__main__":
    # 🔒 Edit these and run the script
    # XML_PATH = "/Users/sean.fite/Desktop/bigstitcher.xml"
    # DROPPED_CSV_PATH = "/Users/sean.fite/Desktop/solver_removed_links.csv"
    # OUT_DIR  = "/Users/sean.fite/Desktop/pairwise_qc_out"

    XML_PATH = "s3://aind-open-data/HCR_831988-s1-ls2_2026-05-27_00-00-00_processed_2026-05-28_01-30-18/image_tile_alignment/bigstitcher.xml"
    DROPPED_CSV_PATH = "s3://aind-open-data/HCR_831988-s1-ls2_2026-05-27_00-00-00_processed_2026-05-28_01-30-18/image_tile_alignment/solver_removed_links.csv"
    OUT_DIR  = "/Users/sean.fite/Desktop/pairwise_qc_out"

    XY_THRESH_LOG2  = 2.0   # same meaning as your original xy_thresh
    BAD_CORR_THRESH = 0.99  # links below this are drawn as 'bad' on the grid

    run_qc(XML_PATH, OUT_DIR, XY_THRESH_LOG2, BAD_CORR_THRESH, DROPPED_CSV_PATH)