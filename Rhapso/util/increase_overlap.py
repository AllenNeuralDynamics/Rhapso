from __future__ import annotations
import re
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

TILE_RE = re.compile(r"Tile_X_(\d+)_Y_(\d+)", re.IGNORECASE)


def indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print indentation for ElementTree output."""
    i = "\n" + level * "\t"
    if len(elem):
        if not (elem.text and elem.text.strip()):
            elem.text = i + "\t"
        for child in elem:
            indent(child, level + 1)
        if not (elem.tail and elem.tail.strip()):
            elem.tail = i
    else:
        if level and not (elem.tail and elem.tail.strip()):
            elem.tail = i


def parse_affine_3x4(text: str) -> List[float]:
    parts = text.strip().split()
    if len(parts) != 12:
        raise ValueError(f"Expected 12 affine values, got {len(parts)}")
    return [float(p) for p in parts]


def format_affine_3x4(vals: List[float]) -> str:
    return " ".join(f"{v:.5f}" for v in vals)


def robust_step_from_pairs(samples: List[Tuple[int, float]]) -> float:
    """
    Estimate the step (slope) between tile index and translation using
    median differences on unique indices.
    samples: [(tile_index, translation_value), ...]
    """
    by_idx: Dict[int, List[float]] = {}
    for idx, val in samples:
        by_idx.setdefault(idx, []).append(val)

    uniq = sorted((idx, statistics.median(vals)) for idx, vals in by_idx.items())
    if len(uniq) < 2:
        raise ValueError("Not enough unique indices to estimate step.")

    diffs: List[float] = []
    for (i0, v0), (i1, v1) in zip(uniq, uniq[1:]):
        di = i1 - i0
        if di == 0:
            continue
        diffs.append((v1 - v0) / di)

    if not diffs:
        raise ValueError("Could not compute step.")
    return statistics.median(diffs)


def pick_anchor_index(indices: List[int]) -> int:
    """Pick median index as anchor."""
    s = sorted(indices)
    return s[len(s) // 2]


@dataclass(frozen=True)
class TileInfo:
    tile_x: int
    tile_y: int
    size_x: int
    size_y: int


def add_nominal_overlap(
    input_xml: str,
    output_xml: str,
    add_overlap: float = 0.05,
    transform_name: str = "Translation to Nominal Grid",
    *,
    anchor_y_mode: str = "median",   # "median" (old behavior), or "min"/"max" to anchor at seam row
    anchor_x_from_seam: bool = False,
) -> None:
    """
    Increase nominal overlap by shrinking the grid step.

    add_overlap: fraction of tile size to add as overlap (0.05 = +5 percentage points).
    anchor_y_mode:
        - "median" (default): keep previous global behavior (anchor at median tile_y)
        - "min": anchor at the row with smallest tile_y (e.g. seam row for a TOP region)
        - "max": anchor at the row with largest tile_y (e.g. seam row for a BOTTOM region)
    anchor_x_from_seam:
        - If True, pick anchor_x as the median tile_x **within the seam row**.
        - If False, use median tile_x over all tiles (old behavior).

    This version supports two cases:

    1) ViewSetup names like "Tile_X_0000_Y_0000_..." (old behavior) – use TILE_RE.
    2) ViewSetup names like "tile_000000_ch_488" with 5×4 nominal translations – in this
       case we ignore the names and infer the (tile_x, tile_y) grid purely from the
       nominal translation values in the ViewRegistrations.
    """
    if not (0.0 < add_overlap < 0.5):
        raise ValueError("add_overlap should be in (0, 0.5)")

    if anchor_y_mode not in ("median", "min", "max"):
        raise ValueError("anchor_y_mode must be 'median', 'min', or 'max'.")

    tree = ET.parse(input_xml)
    root = tree.getroot()

    # ---- 1. Gather per-setup metadata (name, size, etc.)
    setup_meta: Dict[str, Dict[str, object]] = {}
    tile_size_x: Optional[int] = None
    tile_size_y: Optional[int] = None

    for vs in root.findall(".//ViewSetups/ViewSetup"):
        sid = vs.findtext("id")
        name = vs.findtext("name") or ""
        size_txt = vs.findtext("size") or ""

        if not sid or not size_txt.strip():
            continue

        size_parts = size_txt.strip().split()
        if len(size_parts) < 2:
            continue

        sx = int(float(size_parts[0]))
        sy = int(float(size_parts[1]))

        if tile_size_x is None:
            tile_size_x = sx
            tile_size_y = sy

        setup_meta[sid] = {
            "name": name,
            "size_x": sx,
            "size_y": sy,
        }

    if not setup_meta:
        raise RuntimeError("No ViewSetups found (cannot proceed).")

    # ---- 2. Collect all nominal transforms for the given transform_name
    transform_records: List[Dict[str, object]] = []
    for vr in root.findall(".//ViewRegistrations/ViewRegistration"):
        setup = vr.get("setup")
        if not setup or setup not in setup_meta:
            continue

        vt_target: Optional[ET.Element] = None
        for vt in vr.findall("ViewTransform"):
            nm = (vt.findtext("Name") or "").strip()
            if nm == transform_name:
                vt_target = vt
                break
        if vt_target is None:
            continue

        affine_el = vt_target.find("affine")
        if affine_el is None or not (affine_el.text and affine_el.text.strip()):
            continue

        vals = parse_affine_3x4(affine_el.text)
        tx = vals[3]
        ty = vals[7]

        transform_records.append(
            {
                "setup": setup,
                "affine_el": affine_el,
                "vals": vals,
                "tx": tx,
                "ty": ty,
            }
        )

    if len(transform_records) < 2:
        raise RuntimeError("Not enough nominal transforms found to estimate grid steps.")

    # ---- 3. Build setup_infos either from name pattern or inferred grid
    setup_infos: Dict[str, TileInfo] = {}

    # 3a. Try name-based pattern "Tile_X_####_Y_####" (old behavior)
    for sid, meta in setup_meta.items():
        name = meta["name"]  # type: ignore[assignment]
        if not isinstance(name, str):
            continue
        m = TILE_RE.search(name)
        if not m:
            continue

        tile_x = int(m.group(1))
        tile_y = int(m.group(2))
        setup_infos[sid] = TileInfo(
            tile_x=tile_x,
            tile_y=tile_y,
            size_x=int(meta["size_x"]),  # type: ignore[arg-type]
            size_y=int(meta["size_y"]),  # type: ignore[arg-type]
        )

    # 3b. Fallback: infer 2D grid purely from translations (for "tile_000000_ch_488" style)
    if not setup_infos:
        xs = [float(rec["tx"]) for rec in transform_records]
        ys = [float(rec["ty"]) for rec in transform_records]

        def cluster_coords(vals: List[float], tol: float = 1.0) -> List[float]:
            """
            Cluster nearly-identical coordinates (within `tol`) and return
            sorted cluster centroids.

            For your 794495 dataset, Ty within a row differ by << 1 pixel,
            while different rows are ~10k pixels apart, so tol=1.0 is plenty.
            """
            if not vals:
                return []
            vals_sorted = sorted(vals)
            clusters: List[List[float]] = [[vals_sorted[0]]]
            for v in vals_sorted[1:]:
                if abs(v - clusters[-1][-1]) <= tol:
                    clusters[-1].append(v)
                else:
                    clusters.append([v])
            centroids = [statistics.mean(c) for c in clusters]
            centroids.sort()
            return centroids

        unique_tx = cluster_coords(xs, tol=1.0)
        unique_ty = cluster_coords(ys, tol=1.0)

        if len(unique_tx) < 2 or len(unique_ty) < 2:
            raise RuntimeError(
                "Could not infer a 2D tile grid from translations: "
                f"{len(unique_tx)} unique X, {len(unique_ty)} unique Y."
            )

        def nearest_index(v: float, centers: List[float]) -> int:
            best_idx = 0
            best_dist = abs(v - centers[0])
            for i, c in enumerate(centers[1:], start=1):
                d = abs(v - c)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            return best_idx

        # Assign a (tile_x, tile_y) to every setup based on its (tx, ty)
        for rec in transform_records:
            sid = rec["setup"]  # type: ignore[assignment]
            meta = setup_meta[sid]

            col = nearest_index(float(rec["tx"]), unique_tx)
            row = nearest_index(float(rec["ty"]), unique_ty)

            setup_infos[sid] = TileInfo(
                tile_x=col,
                tile_y=row,
                size_x=int(meta["size_x"]),  # type: ignore[arg-type]
                size_y=int(meta["size_y"]),  # type: ignore[arg-type]
            )

    if not setup_infos:
        raise RuntimeError(
            "No tile indices could be derived from either ViewSetup names or translations."
        )

    # Use any TileInfo to get tile size
    any_info = next(iter(setup_infos.values()))
    tile_size_x = any_info.size_x
    tile_size_y = any_info.size_y

    # ---- 4. Build (index, translation) samples and target list
    samples_x: List[Tuple[int, float]] = []
    samples_y: List[Tuple[int, float]] = []
    targets: List[Tuple[ET.Element, List[float], str]] = []

    for rec in transform_records:
        sid = rec["setup"]  # type: ignore[assignment]
        info = setup_infos[sid]
        tx = float(rec["tx"])
        ty = float(rec["ty"])

        samples_x.append((info.tile_x, tx))
        samples_y.append((info.tile_y, ty))
        targets.append((rec["affine_el"], rec["vals"], sid))  # type: ignore[index]

    if len(samples_x) < 2 or len(samples_y) < 2:
        raise RuntimeError("Not enough nominal transforms found to estimate grid steps.")

    # ---- 5. Estimate current steps (same as before)
    old_step_x = robust_step_from_pairs(samples_x)
    old_step_y = robust_step_from_pairs(samples_y)

    # ---- 6. Compute new steps to add overlap
    # new_step = old_step - add_overlap * tile_size
    new_step_x = old_step_x - (add_overlap * tile_size_x)
    new_step_y = old_step_y - (add_overlap * tile_size_y)

    if new_step_x <= 0 or new_step_y <= 0:
        raise RuntimeError(
            f"Computed non-positive new step(s): new_step_x={new_step_x}, new_step_y={new_step_y}. "
            f"Reduce add_overlap."
        )

    # ---- 7. Anchor choice (unchanged semantics, now using inferred tile_x/tile_y)
    all_x = [info.tile_x for info in setup_infos.values()]
    all_y = [info.tile_y for info in setup_infos.values()]

    if anchor_y_mode == "median":
        anchor_y = pick_anchor_index(all_y)
    elif anchor_y_mode == "min":
        anchor_y = min(all_y)
    else:  # "max"
        anchor_y = max(all_y)

    # Y anchor translation = median Ty for that row
    anchor_ty_vals = [ty for (y, ty) in samples_y if y == anchor_y]
    if anchor_ty_vals:
        anchor_ty = statistics.median(anchor_ty_vals)
    else:
        # Fallback to median over all samples
        anchor_ty = statistics.median([ty for _, ty in samples_y])

    # X anchor: either median over seam row or median over all tiles
    if anchor_x_from_seam and anchor_y_mode in ("min", "max"):
        seam_xs = sorted({info.tile_x for info in setup_infos.values() if info.tile_y == anchor_y})
        if seam_xs:
            anchor_x = seam_xs[len(seam_xs) // 2]
        else:
            anchor_x = pick_anchor_index(all_x)
    else:
        anchor_x = pick_anchor_index(all_x)

    anchor_tx_vals = [tx for (x, tx) in samples_x if x == anchor_x]
    if anchor_tx_vals:
        anchor_tx = statistics.median(anchor_tx_vals)
    else:
        anchor_tx = statistics.median([tx for _, tx in samples_x])

    # New intercepts
    a_x_new = anchor_tx - new_step_x * anchor_x
    a_y_new = anchor_ty - new_step_y * anchor_y

    # ---- 8. Rewrite all targets
    updated = 0
    for affine_el, vals, setup in targets:
        info = setup_infos[setup]
        vals[3] = a_x_new + new_step_x * info.tile_x
        vals[7] = a_y_new + new_step_y * info.tile_y
        affine_el.text = format_affine_3x4(vals)
        updated += 1

    indent(root)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)

    # ---- 9. Summary
    old_overlap_x = tile_size_x - old_step_x
    old_overlap_y = tile_size_y - old_step_y
    new_overlap_x = tile_size_x - new_step_x
    new_overlap_y = tile_size_y - new_step_y

    print(f"Wrote: {output_xml}")
    print(f"Updated {updated} transforms named '{transform_name}'")
    print(f"Tile size: X={tile_size_x}px Y={tile_size_y}px")
    print(f"Old step:  X={old_step_x:.3f}px Y={old_step_y:.3f}px")
    print(f"New step:  X={new_step_x:.3f}px Y={new_step_y:.3f}px")
    print(
        f"Old overlap: X={old_overlap_x:.3f}px ({old_overlap_x/tile_size_x:.3%}), "
        f"Y={old_overlap_y:.3f}px ({old_overlap_y/tile_size_y:.3%})"
    )
    print(
        f"New overlap: X={new_overlap_x:.3f}px ({new_overlap_x/tile_size_x:.3%}), "
        f"Y={new_overlap_y/tile_size_y:.3%})"
    )

if __name__ == "__main__":
    INPUT_XML = "/Users/sean.fite/Desktop/dataset_794495.xml"
    OUTPUT_XML = "/Users/sean.fite/Desktop/dataset_794495+7percent.xml"

    # increase percentage points overlap by (e.g. ~5% -> ~10%)
    ADD_OVERLAP = 0.07

    TRANSFORM_NAME = "Translation to Nominal Grid"

    # For a TOP region whose bottom row is the seam, anchor that seam row:
    add_nominal_overlap(
        input_xml=INPUT_XML,
        output_xml=OUTPUT_XML,
        add_overlap=ADD_OVERLAP,
        transform_name=TRANSFORM_NAME,
        anchor_y_mode="min",         # or "max" if your Y indexing goes the other way
        anchor_x_from_seam=True,
    )


# python -m Rhapso.eval.increase_overlap