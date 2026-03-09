import xml.etree.ElementTree as ET

def in_ranges(value: int) -> bool:
    """Return True if integer value is inside any of the KEEP_RANGES."""
    for lo, hi in KEEP_RANGES:
        if lo <= value <= hi:
            return True
    return False


def filter_spimdata_xml(input_path: str, output_path: str) -> None:
    tree = ET.parse(input_path)
    root = tree.getroot()

    # --- SequenceDescription ---
    seq_desc = root.find("SequenceDescription")
    if seq_desc is None:
        raise RuntimeError("No <SequenceDescription> found in XML.")

    # 1) Filter zgroups / zgroup[@setup]
    img_loader = seq_desc.find("ImageLoader")
    if img_loader is not None:
        zgroups = img_loader.find("zgroups")
        if zgroups is not None:
            for zg in list(zgroups):
                if zg.tag != "zgroup":
                    continue
                setup_str = zg.get("setup")
                if not setup_str:
                    continue
                setup_id = int(setup_str)
                if not in_ranges(setup_id):
                    zgroups.remove(zg)

    # 2) Filter ViewSetups/ViewSetup[id]
    view_setups = seq_desc.find("ViewSetups")
    if view_setups is not None:
        # Remove ViewSetup entries whose <id> is not in ranges
        for vs in list(view_setups.findall("ViewSetup")):
            id_elem = vs.find("id")
            if id_elem is None or not id_elem.text:
                continue
            setup_id = int(id_elem.text.strip())
            if not in_ranges(setup_id):
                view_setups.remove(vs)

        # 3) Filter Attributes name="tile" / Tile[id]
        for attrs in list(view_setups.findall("Attributes")):
            if attrs.get("name") == "tile":
                for tile in list(attrs.findall("Tile")):
                    id_elem = tile.find("id")
                    if id_elem is None or not id_elem.text:
                        continue
                    tile_id = int(id_elem.text.strip())
                    if not in_ranges(tile_id):
                        attrs.remove(tile)

    # --- ViewRegistrations / ViewRegistration[@setup] ---
    view_regs = root.find("ViewRegistrations")
    if view_regs is not None:
        for vr in list(view_regs.findall("ViewRegistration")):
            setup_str = vr.get("setup")
            if not setup_str:
                continue
            setup_id = int(setup_str)
            if not in_ranges(setup_id):
                view_regs.remove(vr)

    # Write filtered XML
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Filtered XML written to: {output_path}")


if __name__ == "__main__":
    # Hard-coded ranges (inclusive)
    KEEP_RANGES = [
        (0, 8),
        (9, 27),
        (28, 56),
        (57, 94),
        (101, 138),
        (148, 185),
        (195, 232),
        (242, 279),
        (289, 326),
        (336, 373),
        (383, 420),
        (430, 467),
        (477, 514),
        (524, 561),
        (571, 608),
        (618, 655),
        (665, 702),
        (712, 749),
        (759, 796),
        (806, 843),
        (853, 889),
        (899, 935),
        (945, 981),
        (990, 1026),
        (1032, 1068),
        (1071, 1106),
        (1107, 1140),
        (1141, 1170),
        (1171, 1197),
        (1198, 1219),
        (1220, 1238),
        (1239, 1253),
        (1254, 1262),
    ]

    INPUT_XML = "/Users/sean.fite/Desktop/813720_round_2/HCR_813720_round_2_start.xml"
    OUTPUT_XML = "/Users/sean.fite/Desktop/813720_round_2/Filtered_Region_7/dataset_start.xml"

    filter_spimdata_xml(INPUT_XML, OUTPUT_XML)


    # python -m Rhapso.eval.filter_tiles_from_xml