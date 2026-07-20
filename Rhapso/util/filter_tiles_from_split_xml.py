import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Set


def int_text(elem: ET.Element | None) -> int | None:
    if elem is None or elem.text is None:
        return None
    text = elem.text.strip()
    if text == "":
        return None
    return int(text)


def int_attr(elem: ET.Element, name: str) -> int | None:
    value = elem.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value.strip())


def remove_children_if(parent: ET.Element | None, tag: str, should_remove) -> int:
    if parent is None:
        return 0

    removed = 0
    for child in list(parent.findall(tag)):
        if should_remove(child):
            parent.remove(child)
            removed += 1

    return removed


def find_split_loader(seq_desc: ET.Element) -> ET.Element:
    for img_loader in seq_desc.findall("ImageLoader"):
        if img_loader.get("format") == "split.viewerimgloader":
            return img_loader

    raise RuntimeError("No split ImageLoader found: expected format='split.viewerimgloader'.")


def filter_original_view_setups(
    inner_seq_desc: ET.Element,
    keep_old_tile_ids: Set[int],
) -> dict[str, int]:
    """
    Filters the INNER original ViewSetups section.

    This section still uses original / old tile ids:
      ViewSetup/id == OldId
      zgroup setup == OldId
      Tile/id == OldId
    """
    stats = {
        "removed_inner_view_setups": 0,
        "removed_inner_tile_attrs": 0,
        "removed_inner_missing_views": 0,
    }

    view_setups = inner_seq_desc.find("ViewSetups")
    if view_setups is None:
        raise RuntimeError("No inner <ViewSetups> found.")

    stats["removed_inner_view_setups"] = remove_children_if(
        view_setups,
        "ViewSetup",
        lambda vs: int_text(vs.find("id")) not in keep_old_tile_ids,
    )

    for attrs in view_setups.findall("Attributes"):
        if attrs.get("name") == "tile":
            stats["removed_inner_tile_attrs"] += remove_children_if(
                attrs,
                "Tile",
                lambda tile: int_text(tile.find("id")) not in keep_old_tile_ids,
            )

        elif attrs.get("name") == "illumination":
            stats["removed_inner_tile_attrs"] += remove_children_if(
                attrs,
                "Illumination",
                lambda illum: int_text(illum.find("id")) not in keep_old_tile_ids,
            )

    missing_views = inner_seq_desc.find("MissingViews")
    if missing_views is not None:
        for child in list(missing_views):
            setup_id = int_attr(child, "setup")
            if setup_id is not None and setup_id not in keep_old_tile_ids:
                missing_views.remove(child)
                stats["removed_inner_missing_views"] += 1

    return stats


def filter_split_view_setups(
    outer_seq_desc: ET.Element,
    keep_new_setup_ids: Set[int],
    keep_old_tile_ids: Set[int],
) -> dict[str, int]:
    """
    Filters the TOP-LEVEL split ViewSetups section.

    This section uses split NewId ids:
      ViewSetup/id == NewId
      ViewSetup/attributes/tile == NewId
      Tile/id == NewId

    But illumination id usually maps to old tile:
      ViewSetup/attributes/illumination == OldId
      Illumination/id == OldId
      Illumination/name == old_tile_{OldId}
    """
    stats = {
        "removed_split_view_setups": 0,
        "removed_split_tile_attrs": 0,
        "removed_split_illumination_attrs": 0,
        "removed_split_missing_views": 0,
    }

    view_setups = outer_seq_desc.find("ViewSetups")
    if view_setups is None:
        print("[split-filter] no top-level split <ViewSetups> found; skipping")
        return stats

    # 1) Remove split ViewSetup entries outside kept NewIds.
    stats["removed_split_view_setups"] = remove_children_if(
        view_setups,
        "ViewSetup",
        lambda vs: int_text(vs.find("id")) not in keep_new_setup_ids,
    )

    # 2) Clean top-level Attributes sections.
    for attrs in view_setups.findall("Attributes"):
        name = attrs.get("name")

        if name == "tile":
            # Split Tile ids are NewIds: keep 0..19 for old tile 0.
            stats["removed_split_tile_attrs"] += remove_children_if(
                attrs,
                "Tile",
                lambda tile: int_text(tile.find("id")) not in keep_new_setup_ids,
            )

        elif name == "illumination":
            # Illumination ids are old tile ids: keep only 0 for old tile 0.
            stats["removed_split_illumination_attrs"] += remove_children_if(
                attrs,
                "Illumination",
                lambda illum: int_text(illum.find("id")) not in keep_old_tile_ids,
            )

        # Usually channel and angle stay global, so leave them alone.

    # 3) If top-level MissingViews exists, it likely refers to split NewIds.
    missing_views = outer_seq_desc.find("MissingViews")
    if missing_views is not None:
        for child in list(missing_views):
            setup_id = int_attr(child, "setup")
            if setup_id is not None and setup_id not in keep_new_setup_ids:
                missing_views.remove(child)
                stats["removed_split_missing_views"] += 1

    return stats


def filter_split_xml_to_old_tiles(
    input_path: str,
    output_path: str,
    keep_old_tile_ids: Iterable[int],
) -> None:
    keep_old_tile_ids: Set[int] = set(keep_old_tile_ids)

    tree = ET.parse(input_path)
    root = tree.getroot()

    outer_seq_desc = root.find("SequenceDescription")
    if outer_seq_desc is None:
        raise RuntimeError("No top-level <SequenceDescription> found.")

    split_loader = find_split_loader(outer_seq_desc)

    inner_img_loader = split_loader.find("ImageLoader")
    if inner_img_loader is None:
        raise RuntimeError("No inner <ImageLoader> found inside split loader.")

    inner_seq_desc = split_loader.find("SequenceDescription")
    if inner_seq_desc is None:
        raise RuntimeError("No inner <SequenceDescription> found inside split loader.")

    setup_ids = split_loader.find("SetupIds")
    if setup_ids is None:
        raise RuntimeError("No <SetupIds> found inside split loader.")

    # ---------------------------------------------------------------------
    # Build NewId set from kept OldId values.
    # For old tile 0, this should become NewIds 0..19 in your current file.
    # ---------------------------------------------------------------------
    keep_new_setup_ids: Set[int] = set()

    for setup_def in setup_ids.findall("SetupIdDefinition"):
        new_id = int_text(setup_def.find("NewId"))
        old_id = int_text(setup_def.find("OldId"))

        if new_id is None or old_id is None:
            continue

        if old_id in keep_old_tile_ids:
            keep_new_setup_ids.add(new_id)

    if not keep_new_setup_ids:
        raise RuntimeError(
            f"No split SetupIdDefinition entries found for old tile ids: "
            f"{sorted(keep_old_tile_ids)}"
        )

    print("[split-filter] keep old tile ids:", sorted(keep_old_tile_ids))
    print("[split-filter] keep split new setup ids:", sorted(keep_new_setup_ids))
    print("[split-filter] kept split setup count:", len(keep_new_setup_ids))

    # ---------------------------------------------------------------------
    # Filter SetupIdDefinition by OldId.
    # ---------------------------------------------------------------------
    removed_setup_defs = remove_children_if(
        setup_ids,
        "SetupIdDefinition",
        lambda d: int_text(d.find("OldId")) not in keep_old_tile_ids,
    )

    # ---------------------------------------------------------------------
    # Filter inner zgroups by OldId.
    # ---------------------------------------------------------------------
    zgroups = inner_img_loader.find("zgroups")
    removed_zgroups = remove_children_if(
        zgroups,
        "zgroup",
        lambda zg: int_attr(zg, "setup") not in keep_old_tile_ids,
    )

    # ---------------------------------------------------------------------
    # Filter INNER original ViewSetups by OldId.
    # ---------------------------------------------------------------------
    original_stats = filter_original_view_setups(
        inner_seq_desc=inner_seq_desc,
        keep_old_tile_ids=keep_old_tile_ids,
    )

    # ---------------------------------------------------------------------
    # Filter TOP-LEVEL split ViewSetups by NewId.
    # This is the missing part from the previous script.
    # ---------------------------------------------------------------------
    split_stats = filter_split_view_setups(
        outer_seq_desc=outer_seq_desc,
        keep_new_setup_ids=keep_new_setup_ids,
        keep_old_tile_ids=keep_old_tile_ids,
    )

    # ---------------------------------------------------------------------
    # Filter top-level ViewRegistrations by NewId.
    # ---------------------------------------------------------------------
    view_regs = root.find("ViewRegistrations")
    removed_view_regs = 0

    if view_regs is not None:
        removed_view_regs = remove_children_if(
            view_regs,
            "ViewRegistration",
            lambda vr: int_attr(vr, "setup") not in keep_new_setup_ids,
        )

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print("[split-filter] removed SetupIdDefinition:", removed_setup_defs)
    print("[split-filter] removed inner zgroup:", removed_zgroups)

    for key, value in original_stats.items():
        print(f"[split-filter] {key}: {value}")

    for key, value in split_stats.items():
        print(f"[split-filter] {key}: {value}")

    print("[split-filter] removed top-level ViewRegistration:", removed_view_regs)
    print("[split-filter] wrote:", output_path)


if __name__ == "__main__":
    INPUT_XML = "/Users/sean.fite/Desktop/exaSPIM_720164_split.xml"
    OUTPUT_XML = "/Users/sean.fite/Desktop/split_tile_0_only.xml"

    # Original tile ids to keep.
    KEEP_OLD_TILE_IDS = [0]

    filter_split_xml_to_old_tiles(
        input_path=INPUT_XML,
        output_path=OUTPUT_XML,
        keep_old_tile_ids=KEEP_OLD_TILE_IDS,
    )