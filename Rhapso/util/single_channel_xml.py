#!/usr/bin/env python3
"""
Filter a BigStitcher/SpimData XML down to a single channel (e.g. 488).

Keeps only:
- ImageLoader/zgroups/zgroup entries for the channel
- ViewSetups/ViewSetup entries for the channel
- ViewRegistrations/ViewRegistration entries for kept setups
- ViewSetups/Attributes trimmed so:
    - <Attributes name="channel"> contains only that channel
    - <Attributes name="tile"> contains only kept tiles (ids align with setup ids here)

Notes:
- We DO NOT renumber setup/tile IDs. We keep original IDs so transforms remain valid.
- Channel matching prefers <ViewSetup>/<attributes>/<channel> == channel,
  and falls back to name/path containing "_ch_<channel>."
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def _int_text(el: ET.Element | None) -> int | None:
    t = _text(el)
    if not t:
        return None
    try:
        return int(t)
    except ValueError:
        return None


def _looks_like_channel_in_name(s: str, channel: int) -> bool:
    return f"_ch_{channel}." in s


def collect_keep_setup_ids(root: ET.Element, channel: int) -> set[int]:
    keep: set[int] = set()
    vs = root.find("./SequenceDescription/ViewSetups")
    if vs is None:
        return keep

    for viewsetup in list(vs.findall("./ViewSetup")):
        setup_id = _int_text(viewsetup.find("./id"))
        if setup_id is None:
            continue

        ch_val = _int_text(viewsetup.find("./attributes/channel"))
        name = _text(viewsetup.find("./name"))

        if ch_val == channel or _looks_like_channel_in_name(name, channel):
            keep.add(setup_id)

    return keep


def filter_zgroups(root: ET.Element, keep_setups: set[int], channel: int) -> None:
    zgroups = root.find("./SequenceDescription/ImageLoader/zgroups")
    if zgroups is None:
        return

    for zg in list(zgroups.findall("./zgroup")):
        setup_attr = zg.get("setup")
        setup_id = int(setup_attr) if setup_attr and setup_attr.isdigit() else None
        path_text = _text(zg.find("./path"))

        keep = False
        if setup_id is not None and setup_id in keep_setups:
            keep = True
        elif _looks_like_channel_in_name(path_text, channel):
            keep = True
            if setup_id is not None:
                keep_setups.add(setup_id)

        if not keep:
            zgroups.remove(zg)


def filter_viewsetups_and_attributes(root: ET.Element, keep_setups: set[int], channel: int) -> None:
    vs = root.find("./SequenceDescription/ViewSetups")
    if vs is None:
        return

    # Remove ViewSetup entries not in keep_setups
    for viewsetup in list(vs.findall("./ViewSetup")):
        setup_id = _int_text(viewsetup.find("./id"))
        if setup_id is None or setup_id not in keep_setups:
            vs.remove(viewsetup)

    # Trim attributes blocks
    for attrs in list(vs.findall("./Attributes")):
        name_attr = (attrs.get("name") or "").strip()

        if name_attr == "channel":
            for ch in list(attrs.findall("./Channel")):
                ch_id = _int_text(ch.find("./id"))
                ch_name = _text(ch.find("./name"))
                if ch_id != channel and ch_name != str(channel):
                    attrs.remove(ch)

        elif name_attr == "tile":
            # In these XMLs, tile ids align with setup ids (as shown in your example).
            for tile in list(attrs.findall("./Tile")):
                tid = _int_text(tile.find("./id"))
                if tid is None or tid not in keep_setups:
                    attrs.remove(tile)


def filter_viewregistrations(root: ET.Element, keep_setups: set[int]) -> None:
    vrs = root.find("./ViewRegistrations")
    if vrs is None:
        return

    for vr in list(vrs.findall("./ViewRegistration")):
        setup_attr = vr.get("setup")
        setup_id = int(setup_attr) if setup_attr and setup_attr.isdigit() else None
        if setup_id is None or setup_id not in keep_setups:
            vrs.remove(vr)


def filter_spimdata_channel(input_xml: Path, output_xml: Path, channel: int) -> None:
    tree = ET.parse(input_xml)
    root = tree.getroot()

    keep_setups = collect_keep_setup_ids(root, channel)

    # Fallback: infer from zgroups path strings if needed
    if not keep_setups:
        zgroups = root.find("./SequenceDescription/ImageLoader/zgroups")
        if zgroups is not None:
            for zg in zgroups.findall("./zgroup"):
                path_text = _text(zg.find("./path"))
                if _looks_like_channel_in_name(path_text, channel):
                    setup_attr = zg.get("setup")
                    if setup_attr and setup_attr.isdigit():
                        keep_setups.add(int(setup_attr))

    if not keep_setups:
        raise RuntimeError(
            f"No setups found for channel {channel}. "
            f"Expected <attributes><channel>{channel}</channel> or filenames with '_ch_{channel}.'"
        )

    filter_zgroups(root, keep_setups, channel)
    filter_viewsetups_and_attributes(root, keep_setups, channel)
    filter_viewregistrations(root, keep_setups)

    ET.indent(tree, space="\t", level=0)  # pretty print (py3.9+)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


# -------------------------
# Inline params / "main"
# -------------------------
if __name__ == "__main__":
    INPUT_XML = Path("/Users/sean.fite/Desktop/dataset.xml")
    OUTPUT_XML = Path("/Users/sean.fite/Desktop/dataset_single.xml")
    CHANNEL = 488

    try:
        filter_spimdata_channel(INPUT_XML, OUTPUT_XML, CHANNEL)
        print(f"Wrote: {OUTPUT_XML}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
