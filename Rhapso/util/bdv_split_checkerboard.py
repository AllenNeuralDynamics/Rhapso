import xml.etree.ElementTree as ET


def _collect_tile_data_split_simple(root):
    """
    Split mode (simple, only):

    We want:
      - All sub-tiles (NewId) that came from the same original tile (OldId)
        to share the *same* checkerboard color.
      - The original tiles (OldId = 0,1,2,...) to alternate colors in a
        2D checkerboard pattern.

    For this dataset:
      - There are 5 columns and 4 rows of original tiles (20 total).
      - Ordering is right-to-left, top-to-bottom:
          index 0 = top-right, then move left across the row,
          then next row down, etc.

    We use the OldId index (sorted OldId) as a row-major index
    and map it onto a 5×4 grid:

        idx_1d in [0..19]
        row = idx_1d // 5
        col = idx_1d % 5   (columns 0..4; horizontal flip doesn’t change parity)

    Checkerboard is then based on (col + row) % 2.
    """

    setup_ids_elem = root.find(".//SetupIds")
    if setup_ids_elem is None:
        raise RuntimeError(
            "Split mode: <SetupIds> not found; cannot map sub-tiles back to original tiles."
        )

    new_to_old = {}
    old_ids_set = set()

    for def_el in setup_ids_elem.findall("SetupIdDefinition"):
        new_el = def_el.find("NewId")
        old_el = def_el.find("OldId")
        if new_el is None or old_el is None:
            continue
        if new_el.text is None or old_el.text is None:
            continue
        try:
            new_id = int(new_el.text.strip())
            old_id = int(old_el.text.strip())
        except ValueError:
            continue
        new_to_old[new_id] = old_id
        old_ids_set.add(old_id)

    if not new_to_old:
        raise RuntimeError("Split mode: SetupIds has no valid NewId/OldId pairs.")

    # Original tile ids (these correspond to your zgroups "setup" indices)
    old_ids_sorted = sorted(old_ids_set)
    old_id_to_index = {oid: idx for idx, oid in enumerate(old_ids_sorted)}

    print("\nSplit mode (simple grouping by original tile):")
    print(f"  Original tile ids (OldId) sorted: {old_ids_sorted}")

    # ---- Fixed 5x4 grid for this dataset ----
    num_cols = 5
    num_rows = 4
    num_tiles = len(old_ids_sorted)

    if num_tiles != num_cols * num_rows:
        print(
            f"WARNING: expected {num_cols * num_rows} original tiles for a 5x4 grid, "
            f"but found {num_tiles}. Checkerboard may be off."
        )

    print(f"\nUsing fixed grid shape: {num_rows} rows x {num_cols} cols")
    print("Assuming ordering is right-to-left, top-to-bottom (row-major).")

    # Build tile_data from SetupIds
    tile_data = []
    print("\nSplit mode: assigning 2D indices (col,row) per original tile (OldId):")

    for new_id in sorted(new_to_old.keys()):
        old_id = new_to_old[new_id]
        idx_1d = old_id_to_index.get(old_id, 0)

        # Row-major index mapped to 5x4 grid
        row = idx_1d // num_cols   # 0..3  (top to bottom)
        col = idx_1d % num_cols    # 0..4  (left to right in our grid coords)

        # Note: BDV's visual “right→left” vs “left→right” doesn’t affect parity
        # because num_cols = 5 is odd; flipping columns adds an even offset.

        x_pos = col
        y_pos = row

        debug_name = f"new{new_id}_old{old_id}_idx{idx_1d}_grid({col},{row})"
        tile_data.append((new_id, x_pos, y_pos, debug_name))

        print(
            f"  Setup {new_id:4d} (sub-tile) -> original tile OldId {old_id:4d}, "
            f"idx={idx_1d}, grid=({col},{row})"
        )

    print(f"\nDetected {len(tile_data)} split sub-tiles mapped onto original tiles")
    return tile_data


def generate_settings_file(dataset_xml, output_xml, existing_settings=None):
    GREEN = "-16711936"  # 0xFF00FF00 in signed int
    RED   = "-65536"     # 0xFFFF0000 in signed int

    print(f"Reading dataset XML {dataset_xml}")
    tree = ET.parse(dataset_xml)
    root = tree.getroot()

    # We assume this is already a split-style dataset with SetupIds present.
    tile_data = _collect_tile_data_split_simple(root)

    # Read existing settings if provided to preserve min/max values
    min_val, max_val = "0.0", "30.0"
    if existing_settings:
        try:
            existing_tree = ET.parse(existing_settings)
            first_setup = existing_tree.find(".//ConverterSetup")
            if first_setup is not None:
                min_elem = first_setup.find("min")
                max_elem = first_setup.find("max")
                if min_elem is not None and min_elem.text is not None:
                    min_val = min_elem.text
                if max_elem is not None and max_elem.text is not None:
                    max_val = max_elem.text
        except Exception as e:
            print(f"WARNING: Could not read existing settings: {e}")

    # ===== Build BDV Settings XML =====
    settings_root = ET.Element("Settings")

    viewer_state = ET.SubElement(settings_root, "ViewerState")
    sources = ET.SubElement(viewer_state, "Sources")

    # One <Source> entry per setup
    for _ in range(len(tile_data)):
        source = ET.SubElement(sources, "Source")
        active = ET.SubElement(source, "active")
        active.text = "true"

    source_groups = ET.SubElement(viewer_state, "SourceGroups")
    # Keep the original behavior: up to 10 groups using the first N setups
    for i in range(min(10, len(tile_data))):
        group = ET.SubElement(source_groups, "SourceGroup")
        active = ET.SubElement(group, "active")
        active.text = "true"
        name = ET.SubElement(group, "name")
        name.text = f"group {i+1}"
        group_id = ET.SubElement(group, "id")
        group_id.text = str(tile_data[i][0])  # setup_id

    display_mode = ET.SubElement(viewer_state, "DisplayMode")
    display_mode.text = "fs"

    interpolation = ET.SubElement(viewer_state, "Interpolation")
    interpolation.text = "nearestneighbor"

    current_source = ET.SubElement(viewer_state, "CurrentSource")
    current_source.text = "0"

    current_group = ET.SubElement(viewer_state, "CurrentGroup")
    current_group.text = "0"

    current_timepoint = ET.SubElement(viewer_state, "CurrentTimePoint")
    current_timepoint.text = "0"

    setup_assignments = ET.SubElement(settings_root, "SetupAssignments")
    converter_setups = ET.SubElement(setup_assignments, "ConverterSetups")

    print("\nApplying checkerboard pattern (by original tile index):")

    # Sort by setup_id for reproducibility
    tile_data_sorted = sorted(tile_data, key=lambda x: x[0])

    for setup_id, x_pos, y_pos, name in tile_data_sorted:
        # Checkerboard on index parity:
        # all sub-tiles of the same original tile share x_pos, so same color.
        is_green = (x_pos + y_pos) % 2 == 0
        color_value = GREEN if is_green else RED

        setup = ET.SubElement(converter_setups, "ConverterSetup")

        id_elem = ET.SubElement(setup, "id")
        id_elem.text = str(setup_id)

        min_elem = ET.SubElement(setup, "min")
        min_elem.text = min_val

        max_elem = ET.SubElement(setup, "max")
        max_elem.text = max_val

        color_elem = ET.SubElement(setup, "color")
        color_elem.text = color_value

        group_id = ET.SubElement(setup, "groupId")
        group_id.text = "0"

        color_name = "GREEN" if is_green else "RED"
        print(f"Setup {setup_id:4d} ({name}) -> index={x_pos}, {color_name}")

    # MinMaxGroup boilerplate
    minmax_groups = ET.SubElement(setup_assignments, "MinMaxGroups")
    minmax_group = ET.SubElement(minmax_groups, "MinMaxGroup")

    id_elem = ET.SubElement(minmax_group, "id")
    id_elem.text = "0"

    full_range_min = ET.SubElement(minmax_group, "fullRangeMin")
    full_range_min.text = "-2.147483648E9"

    full_range_max = ET.SubElement(minmax_group, "fullRangeMax")
    full_range_max.text = "2.147483647E9"

    range_min = ET.SubElement(minmax_group, "rangeMin")
    range_min.text = "0.0"

    range_max = ET.SubElement(minmax_group, "rangeMax")
    range_max.text = "65535.0"

    current_min = ET.SubElement(minmax_group, "currentMin")
    current_min.text = "0.0"

    current_max = ET.SubElement(minmax_group, "currentMax")
    current_max.text = "65535.0"

    # ManualSourceTransforms (identity per setup)
    transforms = ET.SubElement(settings_root, "ManualSourceTransforms")
    for _ in range(len(tile_data)):
        transform = ET.SubElement(transforms, "SourceTransform")
        transform.set("type", "affine")
        affine = ET.SubElement(transform, "affine")
        affine.text = "1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0"

    ET.SubElement(settings_root, "Bookmarks")

    out_tree = ET.ElementTree(settings_root)
    out_tree.write(output_xml, encoding='utf-8', xml_declaration=True)

    print("\n" + "=" * 60)
    print(f"Settings file created: {output_xml}")
    print(f"Total setups (sources): {len(tile_data)}")
    print(
        "Colors applied: all sub-tiles from the same original tile share a color; "
        "original tiles alternate along sorted OldId (path list) order."
    )


if __name__ == "__main__":
    input_xml = "/Users/sean.fite/Desktop/exaSPIM-1/rhapso-solver-split-affine.xml"
    output_xml = "/Users/sean.fite/Desktop/bdv_settings.xml"
    existing_settings_xml = None  # or path to an existing BDV settings file

    generate_settings_file(input_xml, output_xml, existing_settings=existing_settings_xml)
