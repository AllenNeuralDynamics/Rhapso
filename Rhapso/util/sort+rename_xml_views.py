"""
Renumber BigStitcher/SpimData XML setup IDs to be contiguous 0..N-1,
and reorder the major per-setup blocks accordingly, while keeping all
associated metadata (zgroups, ViewSetups, Tile attributes, ViewRegistrations,
StitchingResults PairwiseResult refs, etc.) consistent.

Default behavior:
- Collect all setup IDs from <ViewSetups>/<ViewSetup>/<id>.
- Build mapping by sorting old IDs ascending, then mapping to 0..N-1.
  (Example: 1,3,5,...,71 -> 0,1,2,...,35)
- Apply mapping everywhere it matters:
  * <zgroup setup="...">
  * <ViewSetup><id>...</id>
  * <ViewSetup>/<attributes>/<tile>...</tile>
  * <Attributes name="tile">/<Tile>/<id>...</id>
  * <ViewRegistration setup="...">
  * <StitchingResults>/<PairwiseResult view_setup_a="..." view_setup_b="...">
  * (Optionally) other known places with setup-like attributes

- Reorder lists for readability:
  * zgroups sorted by setup
  * ViewSetup blocks sorted by id (only the <ViewSetup> elements; keeps the <Attributes ...> blocks after them)
  * Tile attribute list sorted by id
  * ViewRegistrations sorted by setup
  * PairwiseResult sorted by (a,b)

Edit INPUT_XML / OUTPUT_XML at bottom and run.
"""

from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple


def indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print indentation (ElementTree doesn't do this by default)."""
    i = "\n" + level * "  "
    if len(elem):
        if not (elem.text and elem.text.strip()):
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not (elem.tail and elem.tail.strip()):
            elem.tail = i
    else:
        if level and not (elem.tail and elem.tail.strip()):
            elem.tail = i


def find_child(parent: ET.Element, tag: str) -> Optional[ET.Element]:
    for c in parent:
        if c.tag == tag:
            return c
    return None


def all_children(parent: ET.Element, tag: str) -> List[ET.Element]:
    return [c for c in list(parent) if c.tag == tag]


def text_int(el: ET.Element) -> int:
    if el.text is None:
        raise ValueError(f"Expected integer text in <{el.tag}> but found empty")
    return int(el.text.strip())


def set_text_int(el: ET.Element, v: int) -> None:
    el.text = str(v)


def build_setup_mapping(root: ET.Element) -> Dict[int, int]:
    """
    Build mapping from old setup IDs -> new contiguous IDs (0..N-1),
    based on <SequenceDescription>/<ViewSetups>/<ViewSetup>/<id>.
    """
    seq = root.find("./SequenceDescription")
    if seq is None:
        raise ValueError("Could not find <SequenceDescription>")

    view_setups = seq.find("./ViewSetups")
    if view_setups is None:
        raise ValueError("Could not find <ViewSetups>")

    view_setup_elems = all_children(view_setups, "ViewSetup")
    if not view_setup_elems:
        raise ValueError("No <ViewSetup> elements found")

    old_ids: List[int] = []
    for vs in view_setup_elems:
        id_el = find_child(vs, "id")
        if id_el is None:
            raise ValueError("A <ViewSetup> is missing an <id> child")
        old_ids.append(text_int(id_el))

    unique_sorted = sorted(set(old_ids))
    if len(unique_sorted) != len(old_ids):
        # Allow duplicates but warn; duplicates are usually a problem.
        print(
            f"WARNING: Duplicate <ViewSetup><id> values detected "
            f"({len(old_ids)} entries, {len(unique_sorted)} unique).",
            file=sys.stderr,
        )

    mapping = {old: new for new, old in enumerate(unique_sorted)}
    return mapping


def remap_int_attr(el: ET.Element, attr_name: str, mapping: Dict[int, int]) -> None:
    v = el.get(attr_name)
    if v is None:
        return
    old = int(v)
    if old not in mapping:
        raise ValueError(f"Found {attr_name}='{old}' but it is not in mapping keys")
    el.set(attr_name, str(mapping[old]))


def remap_int_text(el: ET.Element, mapping: Dict[int, int]) -> None:
    if el.text is None:
        return
    t = el.text.strip()
    if not t:
        return
    old = int(t)
    if old not in mapping:
        raise ValueError(f"Found text setup id '{old}' but it is not in mapping keys")
    el.text = str(mapping[old])


def reorder_children(parent: ET.Element, tag: str, key_fn) -> None:
    """Remove all children with tag, reinsert sorted by key_fn."""
    kids = [c for c in list(parent) if c.tag == tag]
    if not kids:
        return
    for c in kids:
        parent.remove(c)
    kids_sorted = sorted(kids, key=key_fn)
    for c in kids_sorted:
        parent.append(c)


def process_xml(input_path: str, output_path: str) -> None:
    tree = ET.parse(input_path)
    root = tree.getroot()

    mapping = build_setup_mapping(root)

    # ---- Remap + reorder ImageLoader zgroups ----
    zgroups = root.find("./SequenceDescription/ImageLoader/zgroups")
    if zgroups is not None:
        for zg in list(zgroups):
            if zg.tag != "zgroup":
                continue
            # setup attribute
            remap_int_attr(zg, "setup", mapping)
        # sort by setup
        reorder_children(zgroups, "zgroup", key_fn=lambda e: int(e.get("setup", "0")))

    # ---- Remap ViewSetups: ViewSetup ids + attributes/tile ----
    view_setups = root.find("./SequenceDescription/ViewSetups")
    if view_setups is None:
        raise ValueError("Could not find <ViewSetups>")

    # ViewSetup blocks first (they are mixed with <Attributes ...> blocks afterward)
    view_setup_elems = [c for c in list(view_setups) if c.tag == "ViewSetup"]

    for vs in view_setup_elems:
        id_el = find_child(vs, "id")
        if id_el is None:
            raise ValueError("A <ViewSetup> is missing an <id> child")
        old_id = text_int(id_el)
        if old_id not in mapping:
            raise ValueError(f"ViewSetup id {old_id} not in mapping keys")
        new_id = mapping[old_id]
        set_text_int(id_el, new_id)

        # also remap <attributes><tile>...</tile> (this is usually equal to setup id)
        attrs_el = find_child(vs, "attributes")
        if attrs_el is not None:
            tile_el = find_child(attrs_el, "tile")
            if tile_el is not None and (tile_el.text and tile_el.text.strip()):
                # some datasets store tile IDs differently; if it's one of the old IDs, remap it
                tile_val = int(tile_el.text.strip())
                if tile_val in mapping:
                    tile_el.text = str(mapping[tile_val])

    # reorder only the <ViewSetup> elements, keeping the <Attributes ...> blocks after them
    # Strategy: remove all ViewSetup elems then insert them before the first non-ViewSetup element.
    non_vs_children = [c for c in list(view_setups) if c.tag != "ViewSetup"]
    for c in view_setup_elems:
        view_setups.remove(c)

    view_setup_sorted = sorted(view_setup_elems, key=lambda e: text_int(find_child(e, "id")))
    # reinsert at the front, preserving the rest
    for idx, vs in enumerate(view_setup_sorted):
        view_setups.insert(idx, vs)

    # ---- Remap Attributes(name="tile")/<Tile>/<id> and reorder Tile entries ----
    tile_attrs = root.findall("./SequenceDescription/ViewSetups/Attributes[@name='tile']")
    for ta in tile_attrs:
        tiles = [c for c in list(ta) if c.tag == "Tile"]
        for t in tiles:
            id_el = find_child(t, "id")
            if id_el is None:
                continue
            old = text_int(id_el)
            if old in mapping:
                set_text_int(id_el, mapping[old])
        # reorder <Tile> elements by their <id>
        # (remove all Tile children, re-add sorted)
        for t in tiles:
            ta.remove(t)
        tiles_sorted = sorted(tiles, key=lambda e: text_int(find_child(e, "id")))
        for t in tiles_sorted:
            ta.append(t)

    # ---- Remap + reorder ViewRegistrations ----
    vregs = root.find("./ViewRegistrations")
    if vregs is not None:
        for vr in list(vregs):
            if vr.tag != "ViewRegistration":
                continue
            remap_int_attr(vr, "setup", mapping)
        reorder_children(vregs, "ViewRegistration", key_fn=lambda e: int(e.get("setup", "0")))

    # ---- Remap StitchingResults PairwiseResult refs + reorder ----
    stitch = root.find("./StitchingResults")
    if stitch is not None:
        for pr in list(stitch):
            if pr.tag != "PairwiseResult":
                continue
            remap_int_attr(pr, "view_setup_a", mapping)
            remap_int_attr(pr, "view_setup_b", mapping)
        reorder_children(
            stitch,
            "PairwiseResult",
            key_fn=lambda e: (int(e.get("view_setup_a", "0")), int(e.get("view_setup_b", "0"))),
        )

    # ---- Optional: other known places that sometimes carry setup-like attrs ----
    # If you later discover another attribute that references setup IDs, add it here.
    # Example:
    # for el in root.findall(".//*[@setupId]"):
    #     remap_int_attr(el, "setupId", mapping)

    # pretty-print + write
    indent(root)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)

    # mapping summary
    old_sorted = sorted(mapping.keys())
    print(f"✅ Wrote: {output_path}")
    print(f"Remapped {len(old_sorted)} setup IDs:")
    print("  " + ", ".join(f"{o}->{mapping[o]}" for o in old_sorted[:12]) + (" ..." if len(old_sorted) > 12 else ""))


# ------------------------
# Inline params (edit me)
# ------------------------
if __name__ == "__main__":
    INPUT_XML = "/Users/sean.fite/Desktop/bigstitcher_kept.xml"
    OUTPUT_XML = "/Users/sean.fite/Desktop/bigstitcher_sorted.xml"
    process_xml(INPUT_XML, OUTPUT_XML)
