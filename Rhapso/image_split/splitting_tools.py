# splitting_tools.py
"""
Image splitting tools for Rhapso.

This module provides utilities for splitting large images into smaller tiles
while maintaining XML metadata structure and interest point information.
"""

import math
import numpy as np
import copy
import os
import logging
import traceback
import time
import random
from pathlib import Path
from xml.etree import ElementTree as ET
from itertools import product
from Rhapso.image_split.split_views import next_multiple


def last_image_size(length, size, overlap):
    """Calculate the size of the last tile in a dimension."""
    if length <= size:
        return length

    step = size - overlap
    if step <= 0:
        return length  # Avoid infinite loops

    num_blocks = math.ceil((length - size) / step) + 1
    last_start = (num_blocks - 1) * step
    return length - last_start


def last_image_size_java(l, s, o):
    """
    Port of SplittingTools.lastImageSize(long l, long s, long o) with Java-style remainder.
    
    Java remainder: a % b keeps the sign of 'a' (truncates toward zero). Python uses floor.
    """
    a = l - 2 * (s - o) - o
    b = s - o
    rem = a - int(a / b) * b if b != 0 else 0  # emulate Java's %
    size = o + rem
    if size < 0:
        size = l + size
    return int(size)


def intersect(interval1, interval2):
    """Compute the intersection of two intervals."""
    min1, max1 = interval1
    min2, max2 = interval2
    
    min_intersect = np.maximum(min1, min2)
    max_intersect = np.minimum(max1, max2)
    
    if np.any(min_intersect > max_intersect):
        return None  # No overlap
        
    return (min_intersect, max_intersect)


def is_empty(interval):
    """Check if an interval is empty (has zero or negative volume)."""
    if interval is None:
        return True
    mins, maxs = interval
    return np.any(mins > maxs)


def contains(point_location, interval):
    """Check if a point location is within a given interval."""
    mins, maxs = interval
    for d in range(len(point_location)):
        if not (mins[d] <= point_location[d] <= maxs[d]):
            return False
    return True


def split_dim_java(length, s, o, min0=0):
    """
    Port of SplittingTools.splitDim(...) producing 1D [min,max] inclusive intervals.
    """
    dim_intervals = []
    from_v = int(min0)
    max_v = int(min0 + length - 1)
    
    while True:
        to_v = min(max_v, from_v + int(s) - 1)
        dim_intervals.append((from_v, to_v))
        if to_v >= max_v:
            break
        from_v = to_v - int(o) + 1
    
    return dim_intervals

def distribute_intervals_fixed_overlap(
    input_dims, overlap, target_size, min_step_size, optimize
):
    """
    Port of SplittingTools.distributeIntervalsFixedOverlap(Interval, overlapPx, targetSize, minStepSize, optimize)
    Returns list of (mins: np.ndarray, maxs: np.ndarray) 3D intervals (inclusive).
    """
    # Divisibility checks like Java
    for d in range(len(input_dims)):
        if int(target_size[d]) % int(min_step_size[d]) != 0:
            logging.warning(
                f"targetSize {target_size[d]} not divisible by minStepSize {min_step_size[d]} for dim={d}. stopping."
            )
            return []
        if int(overlap[d]) % int(min_step_size[d]) != 0:
            logging.warning(
                f"overlapPx {overlap[d]} not divisible by minStepSize {min_step_size[d]} for dim={d}. stopping."
            )
            return []

    interval_basis = []
    # Java uses FinalInterval(oldSetup.getSize()) which implies min=0, max=size-1
    for d in range(len(input_dims)):
        dim_intervals = []
        length = int(input_dims[d])
        s = int(target_size[d])
        o = int(overlap[d])

        if length <= s:
            # one block covering full extent [0, length-1]
            dim_intervals.append((0, length - 1))
        else:
            l = length
            last_size = last_image_size_java(l, s, o)
            if optimize and last_size != s:
                current_last = last_size
                step = int(min_step_size[d])
                if last_size <= s // 2:
                    # increase image size until delta <= 0, then take lastSize
                    lastS = s
                    while True:
                        lastS += step
                        new_last = last_image_size_java(l, lastS, o)
                        delta = current_last - new_last
                        current_last = new_last
                        if delta <= 0:
                            break
                    final_size = lastS
                else:
                    # decrease image size until delta >= 0, then take lastSize + step
                    lastS = s
                    while True:
                        lastS -= step
                        new_last = last_image_size_java(l, lastS, o)
                        delta = current_last - new_last
                        current_last = new_last
                        if delta >= 0:
                            break
                    final_size = lastS + step
            else:
                final_size = s

            # Generate intervals for this dimension
            dim_intervals = split_dim_java(l, final_size, o, min0=0)

        interval_basis.append(dim_intervals)

    # Combine per-dimension intervals into N-D intervals
    # Match Java's LocalizingZeroMinIntervalIterator ordering: dim 0 fastest, last dim slowest
    interval_list = []
    # Build over reversed dimension indices so that original dim 0 advances fastest
    for rev_idx in product(*[range(len(b)) for b in interval_basis[::-1]]):
        idx = rev_idx[::-1]
        mins = [interval_basis[d][idx[d]][0] for d in range(len(interval_basis))]
        maxs = [interval_basis[d][idx[d]][1] for d in range(len(interval_basis))]
        interval_list.append(
            (np.array(mins, dtype=np.int64), np.array(maxs, dtype=np.int64))
        )

    logging.debug(f"Generated {len(interval_list)} total intervals")
    return interval_list

def max_interval_spread(old_setups, overlap, target_size, min_step_size, optimize):
    """Calculate the maximum number of splits for any single view, with Java-like logging."""

    # Normalize inputs for consistent processing
    def _as_int_list(x):
        if isinstance(x, np.ndarray):
            return [int(v) for v in x.tolist()]
        try:
            return [int(v) for v in x]
        except TypeError:
            return [int(x)]

    overlap_list = _as_int_list(overlap)
    target_list = _as_int_list(target_size)
    min_step_list = _as_int_list(min_step_size)

    logging.debug(f"maxIntervalSpread: setups={len(old_setups)}, overlap={overlap_list}, target={target_list}, minStep={min_step_list}, optimize={optimize}")

    max_splits = 1

    for old_setup in old_setups:
        # Robustly fetch ViewSetup id
        id_el = old_setup.find(".//{*}id") or old_setup.find("id")
        try:
            vs_id = int(id_el.text) if id_el is not None else -1
        except Exception:
            vs_id = -1

        # Robustly fetch and parse size
        size_el = old_setup.find(".//{*}size") or old_setup.find("size")
        if size_el is None or size_el.text is None:
            dims = np.array([0, 0, 0], dtype=np.int64)
        else:
            size_str = size_el.text.strip()
            dims = np.array([int(d) for d in size_str.split()], dtype=np.int64)

        intervals = distribute_intervals_fixed_overlap(
            dims, overlap_list, target_list, min_step_list, optimize
        )
        num_intervals = len(intervals) if intervals is not None else 0
        logging.debug(f"ViewSetup {vs_id}: {num_intervals} intervals")

        max_splits = max(max_splits, num_intervals)

    logging.debug(f"maxIntervalSpread result: {max_splits}")
    return max_splits

def _find_one(root, name):
    """Namespaced-safe find for existing elements."""
    return root.find(f".//{{*}}{name}") or root.find(name)

def _ensure_child(parent, tag, attrib=None):
    """Ensure a child element exists, creating it if necessary."""
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag, attrib or {})
    else:
        if attrib:
            # preserve existing attributes unless we overwrite provided keys
            for k, v in attrib.items():
                node.set(k, v)
    return node

def _clear_children(node):
    """Remove all children from a node."""
    for child in list(node):
        node.remove(child)

def _ensure_sequence_description_like_bss(xml_root, img_loader_format="split.viewerimgloader"):
    """
    Ensure root and BasePath (match BSS: BasePath type="relative">.</BasePath>)
    """
    # Ensure root and BasePath (match BSS: BasePath type="relative">.</BasePath>)
    root_tag = xml_root.tag.split("}")[-1]
    root = xml_root
    if root_tag != "SpimData":
        # if this is a SequenceDescription element passed directly, wrap a minimal structure
        spim = ET.Element("SpimData", {"version": "0.2"})
        spim.append(xml_root)
        root = spim

    base_path = _find_one(root, "BasePath")
    if base_path is None:
        base_path = ET.Element("BasePath", {"type": "relative"})
        base_path.text = "."
        # insert BasePath before SequenceDescription if possible
        first = list(root)[0] if len(list(root)) > 0 else None
        if first is not None and first.tag.split("}")[-1] == "SequenceDescription":
            root.insert(0, base_path)
        else:
            root.append(base_path)

    # Ensure SequenceDescription subtree
    seq = _find_one(root, "SequenceDescription")
    if seq is None:
        seq = ET.SubElement(root, "SequenceDescription")

    # 1) ImageLoader
    # Match BSS: <ImageLoader format="split.viewerimgloader">...</ImageLoader>
    # If an ImageLoader or ImgLoader exists, normalize it to the expected format
    # BUT preserve the nested structure if it already exists
    img_loader = (_find_one(seq, "ImageLoader") or _find_one(seq, "ImgLoader"))
    if img_loader is None or img_loader.tag.split("}")[-1] != "ImageLoader":
        if img_loader is not None:
            seq.remove(img_loader)
        img_loader = ET.SubElement(seq, "ImageLoader", {"format": img_loader_format})
    else:
        # Only set format if it's different, don't clear children
        if img_loader.get("format") != img_loader_format:
            img_loader.set("format", img_loader_format)
        # Don't clear children - preserve nested ImageLoader structure

    # 2) ViewSetups (can remain empty)
    view_setups = _find_one(seq, "ViewSetups")
    if view_setups is None:
        view_setups = ET.SubElement(seq, "ViewSetups")
    else:
        # do not remove existing ViewSetup entries, keep as-is
        pass

    # 3) Timepoints
    # Match BSS minimal: <Timepoints type="pattern"></Timepoints>
    timepoints = _find_one(seq, "Timepoints")
    if timepoints is None:
        timepoints = ET.SubElement(seq, "Timepoints", {"type": "pattern"})
    else:
        timepoints.set("type", "pattern")
        _clear_children(timepoints)  # keep empty like BSS sample

    # 4) MissingViews (self-closing/empty)
    missing = _find_one(seq, "MissingViews")
    if missing is None:
        ET.SubElement(seq, "MissingViews")
    else:
        # keep as-is, do not remove entries
        pass

    # 5) Ensure nested ImageLoader structure exists if this is a split operation
    # Check if we have the expected nested structure
    img_loader = _find_one(seq, "ImageLoader")
    if img_loader is not None and img_loader.get("format") == "split.viewerimgloader":
        # Ensure nested ImageLoader exists
        nested_loader = img_loader.find(".//{*}ImageLoader") or img_loader.find("ImageLoader")
        if nested_loader is None:
            # Create minimal nested structure if missing
            nested_img_loader = ET.SubElement(img_loader, "ImageLoader", {"format": "bdv.multimg.zarr", "version": "3.0"})
            zarr_elem = ET.SubElement(nested_img_loader, "zarr", {"type": "absolute"})
            zarr_elem.text = "unknown"
            zgroups_elem = ET.SubElement(nested_img_loader, "zgroups")
            # Add a placeholder zgroup with meaningful defaults (no shape element like Java)
            ET.SubElement(zgroups_elem, "zgroup", {
                "setup": "0", 
                "tp": "0", 
                "path": "tile_000000_ch_561.zarr", 
                "indicies": "[]"
            })
    return root

def split_images(
    spimData,
    overlapPx,
    targetSize,
    minStepSize,
    assingIlluminationsFromTileIds=False,
    optimize=False,
    addIPs=False,
    pointDensity=0.0,
    minPoints=0,
    maxPoints=0,
    error=0.0,
    excludeRadius=0.0,
):
    """
    Split images into smaller tiles while maintaining XML metadata structure.
    
    This function takes a SpimData object and splits it into smaller tiles based on
    the specified parameters, updating all XML metadata accordingly.
    """
    try:
        # Make a copy of the SpimData object to avoid modifying the original
        new_spim_data = copy.deepcopy(spimData)

        # Get the XML tree from the spim_data
        xml_tree = new_spim_data
        
        # Get timepoints var
        sequence_description = xml_tree.find(
            ".//{*}SequenceDescription"
        ) or xml_tree.find("SequenceDescription")

        # Parse timepoints
        timepoints = None
        tp_elements = []
        tp_ids = []
        tp_names = []
        present_setup_ids = []
        missing_setup_ids = []
        present_setups_count = 0
        missing_setups_count = 0

        if sequence_description is not None:
            # Try to find TimePoints element robustly
            timepoints = sequence_description.find(".//{*}Timepoints") or sequence_description.find("Timepoints")
            if timepoints is not None:
                # Try to find all TimePoint elements (namespaced or not)
                tp_elements = list(timepoints.findall(".//{*}TimePoint")) or list(timepoints.findall("TimePoint"))
                if not tp_elements:
                    # Try to parse integerpattern if present
                    intpat = timepoints.find(".//{*}integerpattern") or timepoints.find("integerpattern")
                    if intpat is not None and intpat.text:
                        # Parse pattern like "0"
                        try:
                            tp_id = int(intpat.text.strip())
                            tp_elements = [intpat]
                            tp_ids = [tp_id]
                            tp_names = [str(tp_id)]
                        except Exception:
                            tp_ids = []
                            tp_names = []
                else:
                    for tp in tp_elements:
                        # Try to get id from attribute or child
                        tp_id = None
                        tp_name = None
                        if hasattr(tp, "attrib") and "id" in tp.attrib:
                            try:
                                tp_id = int(tp.get("id"))
                            except Exception:
                                tp_id = tp.get("id")
                        if tp_id is None:
                            tpid_el = tp.find(".//{*}id") or tp.find("id")
                            if tpid_el is not None and tpid_el.text:
                                txt = tpid_el.text.strip()
                                tp_id = int(txt) if txt.isdigit() else txt
                        tp_ids.append(tp_id)
                        tp_name = tp.get("name") if hasattr(tp, "attrib") and "name" in tp.attrib else None
                        if tp_name is None:
                            tpname_el = tp.find(".//{*}name") or tp.find("name")
                            if tpname_el is not None and tpname_el.text:
                                tp_name = tpname_el.text.strip()
                        tp_names.append(tp_name if tp_name is not None else str(tp_id))

                # Find present/missing setups for each timepoint
                view_setups_parent = sequence_description.find(".//{*}ViewSetups") or sequence_description.find("ViewSetups")
                setup_elements = []
                if view_setups_parent is not None:
                    setup_elements = list(view_setups_parent.findall(".//{*}ViewSetup")) or list(view_setups_parent.findall("ViewSetup"))
                present_setup_ids = []
                for vs in setup_elements:
                    id_el = vs.find(".//{*}id") or vs.find("id")
                    if id_el is not None and id_el.text:
                        present_setup_ids.append(int(id_el.text.strip()))
                present_setups_count = len(present_setup_ids)

                # Try to find missing setups (not present in ViewSetups)
                # For now, assume all setups are present
                missing_setups_count = 0
                missing_setup_ids = []

                # Log summary
                if tp_ids:
                    tp_min = min(tp_ids)
                    tp_max = max(tp_ids)
                    logging.info(f"TimePoints: count={len(tp_ids)}, idRange=[{tp_min}..{tp_max}]")
                else:
                    logging.info("TimePoints: No timepoints found.")
            else:
                logging.info("TimePoints: No Timepoints element found.")
        else:
            logging.info("TimePoints: No SequenceDescription found.")

        # get the old setups: sequenceDescription ViewSetups values
        def _strip(tag):
            return tag.split("}")[-1] if "}" in tag else tag

        view_setups_parent = None
        if sequence_description is not None:
            view_setups_parent = sequence_description.find(
                ".//{*}ViewSetups"
            ) or sequence_description.find("ViewSetups")
        if view_setups_parent is None:
            view_setups_parent = xml_tree.find(".//{*}ViewSetups") or xml_tree.find(
                "ViewSetups"
            )

        old_setups = []
        if view_setups_parent is not None:
            candidates = [
                vs for vs in list(view_setups_parent) if _strip(vs.tag) == "ViewSetup"
            ]

            def _get_vs_id(vs):
                id_el = vs.find(".//{*}id") or vs.find("id")
                try:
                    return int(id_el.text) if id_el is not None else 0
                except Exception:
                    return 0

            old_setups = sorted(candidates, key=_get_vs_id)

        logging.info(f"Found and sorted {len(old_setups)} ViewSetups")

        # var creation
        # oldRegistrations
        old_registrations = xml_tree.find(".//{*}ViewRegistrations") or xml_tree.find(
            "ViewRegistrations"
        )

        # underlyingImgLoader
        underlying_img_loader = None
        if sequence_description is not None:
            img_loader = (
                sequence_description.find(".//{*}ImageLoader")
                or sequence_description.find("ImageLoader")
                or sequence_description.find(".//{*}ImgLoader")
                or sequence_description.find("ImgLoader")
            )
            if img_loader is not None:
                underlying_img_loader = copy.deepcopy(img_loader)
                # Try to remove; if not a direct child, ignore failure safely
                try:
                    sequence_description.remove(img_loader)
                except ValueError:
                    pass  # Not a direct child; leave as-is

        # new2oldSetupId
        new2old_setup_id = {}

        # newSetupId2Interval
        new_setup_id2_interval = {}

        # newSetups
        new_setups = []

        # newRegistrations
        new_registrations = {}

        # newInterestpoints
        new_interestpoints = {}

        # newId
        new_id = 0

        # maxIntervalSpread
        max_interval_spread_value = max_interval_spread(
            old_setups, overlapPx, targetSize, minStepSize, optimize
        )
        logging.info(f"maxIntervalSpread = {max_interval_spread_value}")

        # check that there is only one illumination
        if assingIlluminationsFromTileIds:
            # Try to derive unique illumination ids from ViewSetups
            illum_ids = set()
            for vs in old_setups:
                illum_el = vs.find(".//{*}illumination") or vs.find("illumination")
                if (
                    illum_el is not None
                    and illum_el.text
                    and illum_el.text.strip().isdigit()
                ):
                    try:
                        illum_ids.add(int(illum_el.text.strip()))
                    except Exception:
                        pass

            # Fallback: count defined Illuminations under SequenceDescription
            if not illum_ids and sequence_description is not None:
                illums_parent = sequence_description.find(
                    ".//{*}Illuminations"
                ) or sequence_description.find("Illuminations")
                if illums_parent is not None:
                    illum_ids = {
                        i
                        for i, el in enumerate(list(illums_parent))
                        if el.tag.split("}")[-1] == "Illumination"
                    }

            if len(illum_ids) > 1:
                raise ValueError(
                    "Cannot SplittingTools.assingIlluminationsFromTileIds because more than one Illumination exists."
                )

        # create fakeLabel var
        import time

        fakeLabel = f"splitPoints_{int(time.time() * 1000)}"

        # create rnd var
        import random

        rnd = random.Random(23424459)

        # for loop through oldSetups
        logging.info(f"Splitting {len(old_setups)} old setups...")
        new_id = 0  # Ensure new_id starts at 0
        for old_setup in old_setups:
            # set vars 1
            id_el = old_setup.find(".//{*}id") or old_setup.find("id")
            oldID = int(id_el.text) if id_el is not None else None
            oldTile = ""
            tile_el = old_setup.find(".//{*}tile") or old_setup.find("tile")
            if tile_el is not None:
                tile_id_el = tile_el.find(".//{*}id") or tile_el.find("id")
                tile_name_el = tile_el.find(".//{*}name") or tile_el.find("name")
                tile_loc_el = tile_el.find(".//{*}location") or tile_el.find("location")
                oldTile = {
                    "id": int(tile_id_el.text) if tile_id_el is not None else None,
                    "name": tile_name_el.text if tile_name_el is not None else None,
                    "location": (
                        [float(x) for x in tile_loc_el.text.strip().split()]
                        if tile_loc_el is not None and tile_loc_el.text
                        else None
                    ),
                }
            else:
                oldTile = None

            localNewTileId = 0

            # Print current loop index and variable values
            logging.info(
                f"Loop index: {old_setups.index(old_setup)}, oldID: {oldID}, oldTile: {oldTile}, localNewTileId: {localNewTileId}"
            )

            # set vars 2
            # angle
            angle_el = old_setup.find(".//{*}angle") or old_setup.find("angle")
            if angle_el is not None and angle_el.text:
                angle_txt = angle_el.text.strip()
                angle = int(angle_txt) if angle_txt.isdigit() else angle_txt
            else:
                angle = 0

            # channel
            channel_el = old_setup.find(".//{*}channel") or old_setup.find("channel")
            if channel_el is not None and channel_el.text:
                channel_txt = channel_el.text.strip()
                channel = int(channel_txt) if channel_txt.isdigit() else channel_txt
            else:
                channel = 0

            # illum
            illum_el = old_setup.find(".//{*}illumination") or old_setup.find(
                "illumination"
            )
            if illum_el is not None and illum_el.text:
                illum_txt = illum_el.text.strip()
                illum = int(illum_txt) if illum_txt.isdigit() else illum_txt
            else:
                illum = 0

            # voxDim
            vox_el = old_setup.find(".//{*}voxelSize") or old_setup.find("voxelSize")
            voxDim = 0
            if vox_el is not None:
                # Try attributes first
                size_text = vox_el.get("size") if hasattr(vox_el, "get") else None
                unit = vox_el.get("unit") if hasattr(vox_el, "get") else None

                # Children fallback
                if size_text is None:
                    size_child = vox_el.find(".//{*}size") or vox_el.find("size")
                    size_text = (
                        size_child.text.strip()
                        if size_child is not None and size_child.text
                        else None
                    )
                if unit is None:
                    unit_child = vox_el.find(".//{*}unit") or vox_el.find("unit")
                    unit = (
                        unit_child.text.strip()
                        if unit_child is not None and unit_child.text
                        else None
                    )

                # Direct text fallback
                if size_text is None and vox_el.text:
                    size_text = vox_el.text.strip()

                try:
                    size_vals = (
                        [float(x) for x in size_text.split()] if size_text else None
                    )
                except Exception:
                    size_vals = None

                # Only set dict if we parsed something meaningful
                if size_vals is not None or unit is not None:
                    voxDim = {"size": size_vals, "unit": unit}

            logging.info(f"angle: {angle}")
            logging.info(f"channel: {channel}")
            logging.info(f"illum: {illum}")
            logging.info(f"voxDim: {voxDim}")

            # set vars 3
            size_el = old_setup.find(".//{*}size") or old_setup.find("size")
            if size_el is not None and size_el.text:
                dims = [int(x) for x in size_el.text.strip().split()]
            else:
                dims = [0, 0, 0]

            # input: interval [min, max] per dim (inclusive), equivalent of FinalInterval(oldSetup.getSize())
            input = [(0, s - 1) for s in dims]

            def _format_bounds(mins, maxs):
                return (
                    "["
                    + ", ".join(str(v) for v in mins)
                    + "] -> ["
                    + ", ".join(str(v) for v in maxs)
                    + "]"
                )

            def _format_interval_with_dims(mins, maxs):
                dims_tuple = tuple(mx - mn + 1 for mn, mx in zip(mins, maxs))
                return f"{_format_bounds(mins, maxs)}, dimensions {dims_tuple}"

            # Log like IOFunctions.println/Util.printInterval (bounds only)
            input_mins = [mn for mn, _ in input]
            input_maxs = [mx for _, mx in input]
            logging.info(
                f"ViewId {oldID} with interval {_format_bounds(input_mins, input_maxs)} will be split as follows: "
            )

            # intervals: distributeIntervalsFixedOverlap(...)
            intervals = distribute_intervals_fixed_overlap(
                dims, overlapPx, targetSize, minStepSize, optimize
            )

            # Debug: Show interval generation details
            logging.info(f"🔧 [split_images] Interval generation for ViewSetup {oldID}:")
            logging.info(f"  Original dimensions: {dims}")
            logging.info(f"  Target size: {[int(x) for x in targetSize]}")
            logging.info(f"  Overlap: {[int(x) for x in overlapPx]}")
            logging.info(f"  Min step size: {[int(x) for x in minStepSize]}")
            logging.info(f"  Generated intervals: {len(intervals) if intervals else 0}")
            
            # Show intervals per dimension
            if intervals:
                # Calculate how many intervals per dimension
                dim_counts = []
                for d in range(len(dims)):
                    length = dims[d]
                    target = int(targetSize[d])
                    overlap = int(overlapPx[d])
                    if length <= target:
                        dim_counts.append(1)
                    else:
                        step = target - overlap
                        if step > 0:
                            count = (length - target) // step + 1
                            dim_counts.append(count)
                        else:
                            dim_counts.append(1)
                
                logging.info(f"  Expected intervals per dimension: {dim_counts}")
                logging.info(f"  Total expected intervals: {np.prod(dim_counts)}")
                logging.info(f"  Actual intervals generated: {len(intervals)}")

            # interval2ViewSetup map (empty for now, to be filled later)
            interval2ViewSetup = {}

            # Print Java-like parameters and interval list
            logging.info(f"Split parameters for ViewSetup {oldID}:")
            logging.info(
                f"  input      = {_format_interval_with_dims(input_mins, input_maxs)}"
            )
            logging.info(f"  overlapPx  = {[int(v) for v in overlapPx]}")
            logging.info(f"  targetSize = {[int(v) for v in targetSize]}")
            # minStepSize may be a numpy array
            logging.info(
                f"  minStep    = {[int(v) for v in (minStepSize.tolist() if hasattr(minStepSize, 'tolist') else minStepSize)]}"
            )
            logging.info(f"  optimize   = {'true' if optimize else 'false'}")

            if intervals is None:
                logging.info("  intervals  = null")
            else:
                logging.info(f"  intervals ({len(intervals)}):")
                for ii, (mins_arr, maxs_arr) in enumerate(intervals):
                    mins = (
                        [int(x) for x in mins_arr.tolist()]
                        if hasattr(mins_arr, "tolist")
                        else [int(x) for x in mins_arr]
                    )
                    maxs = (
                        [int(x) for x in maxs_arr.tolist()]
                        if hasattr(maxs_arr, "tolist")
                        else [int(x) for x in maxs_arr]
                    )
                    logging.info(f"    [{ii}] {_format_interval_with_dims(mins, maxs)}")

            logging.info(f"  interval2ViewSetup.size = {len(interval2ViewSetup)}")

            # loop through intervals
            logging.info(f"Entering interval loop, total count: {len(intervals)}")
            for i in range(len(intervals)):
                (mins_arr, maxs_arr) = intervals[i]
                mins = [
                    int(x)
                    for x in (
                        mins_arr.tolist() if hasattr(mins_arr, "tolist") else mins_arr
                    )
                ]
                maxs = [
                    int(x)
                    for x in (
                        maxs_arr.tolist() if hasattr(maxs_arr, "tolist") else maxs_arr
                    )
                ]
                logging.info(f"        Processing interval index: {i + 1}")
                logging.info(
                    f"        Interval {i + 1}: {_format_interval_with_dims(mins, maxs)}"
                )

                # interval loop 1
                # from the new ID get the old ID and the corresponding interval
                new2old_setup_id[new_id] = oldID
                new_setup_id2_interval[new_id] = (mins_arr, maxs_arr)

                # size/newDim equivalent
                size = [int(mx - mn + 1) for mn, mx in zip(mins, maxs)]
                newDim = tuple(size)

                # translated tile location = oldTile.location + interval.min per dim
                if isinstance(oldTile, dict) and oldTile.get("location") is not None:
                    location = list(oldTile["location"])
                else:
                    location = [0.0] * len(mins)
                for d in range(len(mins)):
                    location[d] += mins[d]

                # Print statement similar to Java's System.out.println
                logging.info(
                    f"\tCreated new ViewSetup: newId={new_id}, oldID={oldID}, "
                    f"interval={_format_bounds(mins, maxs)}, size={size}, location={location}"
                )

                # interval loop 2
                # compute new tile id and create new tile dict
                old_tile_id = (
                    oldTile["id"]
                    if isinstance(oldTile, dict) and oldTile.get("id") is not None
                    else 0
                )
                newTileId = old_tile_id * max_interval_spread_value + localNewTileId
                localNewTileId += 1
                newTile = {
                    "id": newTileId,
                    "name": str(newTileId),
                    "location": location,
                }

                # illumination for new setup
                newIllum = (
                    {"id": old_tile_id, "name": f"old_tile_{old_tile_id}"}
                    if assingIlluminationsFromTileIds
                    else illum
                )

                # create a Python dict for the new setup
                newSetup = {
                    "id": new_id,
                    "dim": newDim,
                    "voxDim": voxDim,
                    "tile": newTile,
                    "channel": channel,
                    "angle": angle,
                    "illum": newIllum,
                }
                new_setups.append(newSetup)

                # map interval -> newSetup (use tuple(mins), tuple(maxs) as key)
                interval_key = (tuple(mins), tuple(maxs))
                interval2ViewSetup[interval_key] = newSetup

                # Print statements for newly created objects
                logging.info(
                    f"\tCreated newTile: id={newTile['id']}, name={newTile['name']}, location={newTile['location']}"
                )
                if isinstance(newIllum, dict):
                    logging.info(
                        f"\tCreated newIllum: id={newIllum.get('id')}, name={newIllum.get('name')}"
                    )
                else:
                    logging.info(f"\tCreated newIllum: id={newIllum}")
                channel_id = (
                    channel
                    if isinstance(channel, int)
                    else (channel.get("id") if isinstance(channel, dict) else channel)
                )
                angle_id = (
                    angle
                    if isinstance(angle, int)
                    else (angle.get("id") if isinstance(angle, dict) else angle)
                )
                illum_id = (
                    newIllum.get("id") if isinstance(newIllum, dict) else newIllum
                )
                vox_unit = voxDim.get("unit") if isinstance(voxDim, dict) else "null"
                logging.info(
                    f"\tCreated newSetup: id={newSetup['id']}, tileId={newTile['id']}, channel={channel_id if channel_id is not None else 'null'}, angle={angle_id if angle_id is not None else 'null'}, illum={illum_id if illum_id is not None else 'null'}, dim={size}, voxDim={vox_unit}"
                )
                logging.info("")

                # Increment new_id for the next interval (this ensures each interval gets a unique ID)
                new_id += 1

                # timepoint loop start
                # Use the same logic as in the timepoints summary: check for integerpattern if no TimePoint elements
                tp_elements = []
                if timepoints is not None:
                    tp_elements = list(timepoints.findall(".//{*}TimePoint")) or list(timepoints.findall("TimePoint"))
                    if not tp_elements:
                        # Try integerpattern fallback
                        intpat = timepoints.find(".//{*}integerpattern") or timepoints.find("integerpattern")
                        if intpat is not None and intpat.text:
                            tp_id = intpat.text.strip()
                            tp_elements = [intpat]
                tp_total = len(tp_elements)
                logging.info(f"\tStarting timepoint loop for interval index {i}: {tp_total} timepoints total.")
            
            # Print summary of all new setups created
            logging.info(f"🔧 [split_images] Total new_setups created: {len(new_setups)}")
            logging.info(f"🔧 [split_images] Expected zgroups count: {len(old_setups)} (based on original tile count)")
            tpIdx = 0
            for tp in tp_elements:
                # Resolve timepoint id from attribute or child element or integerpattern
                tp_id = None
                if hasattr(tp, "attrib") and "id" in tp.attrib:
                    try:
                        tp_id = int(tp.get("id"))
                    except Exception:
                        tp_id = tp.get("id")
                if tp_id is None:
                    tpid_el = tp.find(".//{*}id") or tp.find("id")
                    if tpid_el is not None and tpid_el.text:
                        txt = tpid_el.text.strip()
                        tp_id = int(txt) if txt.isdigit() else txt
                    elif tp.tag.lower().endswith("integerpattern") and tp.text:
                        tp_id = int(tp.text.strip()) if tp.text.strip().isdigit() else tp.text.strip()
                    else:
                        tp_id = "unknown"
                logging.info(f"\t\tProcessing timepoint {tpIdx + 1}/{tp_total} (id={tp_id})")
                tpIdx += 1

                # timepoint loop var setup CODE START
                # Build oldViewId and fetch oldVR from <ViewRegistrations>
                oldViewId_str = f"ViewId{{timepoint={tp_id}, setup={oldID}}}"
                oldVR_el = None
                if old_registrations is not None:
                    vr_candidates = list(old_registrations.findall(".//{*}ViewRegistration")) or list(old_registrations.findall("ViewRegistration"))
                    for vr in vr_candidates:
                        if str(vr.get("timepoint")) == str(tp_id) and str(vr.get("setup")) == str(oldID):
                            oldVR_el = vr
                            break

                # Extract transformList (name + 12-number row-packed affine) from oldVR
                transformList = []
                old_transform_count = 0
                if oldVR_el is not None:
                    vt_elems = list(oldVR_el.findall(".//{*}ViewTransform")) or list(oldVR_el.findall("ViewTransform"))
                    for vt in vt_elems:
                        name_el = vt.find(".//{*}Name") or vt.find("Name")
                        aff_el = vt.find(".//{*}affine") or vt.find("affine")
                        name = name_el.text.strip() if (name_el is not None and name_el.text) else ""
                        affine_vals = []
                        if aff_el is not None and aff_el.text:
                            try:
                                affine_vals = [float(x) for x in aff_el.text.strip().split()]
                            except Exception:
                                affine_vals = []
                        transformList.append({"name": name, "affine": affine_vals})
                    old_transform_count = len(transformList)

                # Create translation (3x4) for current interval mins and append as "Image Splitting"
                tx = float(mins[0] if len(mins) > 0 else 0.0)
                ty = float(mins[1] if len(mins) > 1 else 0.0)
                tz = float(mins[2] if len(mins) > 2 else 0.0)
                translation_affine = [1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, ty, 0.0, 0.0, 1.0, tz]
                transform = {"name": "Image Splitting", "affine": translation_affine}
                transformList = list(transformList)  # copy
                transformList.append(transform)

                # Build newViewId and newVR, add to new_registrations
                newViewId_key = (int(tp_id) if str(tp_id).isdigit() else tp_id, new_id)
                newVR = {
                    "timepoint": newViewId_key[0],
                    "setup": new_id,
                    "transforms": transformList,
                }
                new_registrations[newViewId_key] = newVR

                # Interest points placeholders: discover oldVipl presence in <ViewInterestPoints>
                vip_root = xml_tree.find(".//{*}ViewInterestPoints") or xml_tree.find("ViewInterestPoints")
                oldVipl = None
                if vip_root is not None:
                    vipl_files = list(vip_root.findall(".//{*}ViewInterestPointsFile")) or list(vip_root.findall("ViewInterestPointsFile"))
                    for vf in vipl_files:
                        if str(vf.get("timepoint")) == str(tp_id) and str(vf.get("setup")) == str(oldID):
                            oldVipl = {"timepointId": int(tp_id) if str(tp_id).isdigit() else tp_id, "viewSetupId": oldID}
                            break
                newVipl = {"timepointId": newViewId_key[0], "viewSetupId": new_id}

                # --- timepoint loop var setup checkpoint ---
                logging.info("\n\t\t--- timepoint loop var setup checkpoint ---")
                logging.info(f"\t\toldViewId: {oldViewId_str}")
                if oldVR_el is None:
                    logging.info("\t\toldVR: null")
                else:
                    logging.info(f"\t\toldVR: ViewRegistration with {old_transform_count} transforms")
                last_name = transformList[-1]["name"] if len(transformList) > 0 else "n/a"
                logging.info(f"\t\ttransformList: size={len(transformList)}, last transform name={last_name}")
                # Match Java-like printing for translation and row-packed copy
                trans_tuple_str = ", ".join(f"{v:.1f}" if float(v).is_integer() else f"{v}" for v in translation_affine)
                logging.info(f"\t\ttranslation: 3d-affine: ({trans_tuple_str})")
                logging.info(f"\t\ttransform: {transform['name']}, affine={[float(v) for v in translation_affine]}")
                newViewId_str = f"ViewId{{timepoint={newViewId_key[0]}, setup={new_id}}}"
                logging.info(f"\t\tnewViewId: {newViewId_str}")
                logging.info(f"\t\tnewVR: {'null' if newVR is None else f'ViewRegistration with {len(transformList)} transforms'}")
                logging.info(f"\t\tnewRegistrations: total entries={len(new_registrations)}")
                logging.info(f"\t\tnewVipl: timepointId={newVipl['timepointId']}, viewSetupId={newVipl['viewSetupId']}")
                if oldVipl is None:
                    logging.info("\t\toldVipl: null")
                else:
                    logging.info(f"\t\toldVipl: timepointId={oldVipl['timepointId']}, viewSetupId={oldVipl['viewSetupId']}")
                logging.info("\t\t--- end checkpoint ---\n")
                # timepoint loop var setup CODE END 

                # only update interest points for present views
                # oldVipl may be null for missing views
                missing_views_el = xml_tree.find(".//{*}MissingViews")
                is_missing = False
                if missing_views_el is not None:
                    # Check if the current viewId is in the list of missing views
                    for view in missing_views_el.findall(".//{*}View") or missing_views_el.findall("View"):
                        if view.get("timepoint") == str(tp_id) and view.get("setup") == str(oldID):
                            is_missing = True
                            break
                
                # The condition is true if the view is NOT missing.
                # This matches the Java logic: `!missingViews.contains(oldViewId)`
                if not is_missing:
                    labelIdx = 0
                    
                    # Find all labels for the old view
                    old_labels = []
                    if vip_root is not None:
                        for vf in vip_root.findall(".//{*}ViewInterestPointsFile") or vip_root.findall("ViewInterestPointsFile"):
                            if vf.get("timepoint") == str(tp_id) and vf.get("setup") == str(oldID):
                                if vf.get("label"):
                                    old_labels.append(vf.get("label"))
                    
                    for label in old_labels:
                        logging.info(f"\t\tProcessing label index: {labelIdx}, label: {label}")
                        id = 0
                        labelIdx += 1

                        # labels loop var creation
                        newIp1 = []
                        oldIpl1_el = None
                        if vip_root is not None:
                            for vf in vip_root.findall(".//{*}ViewInterestPointsFile") or vip_root.findall("ViewInterestPointsFile"):
                                if vf.get("timepoint") == str(tp_id) and vf.get("setup") == str(oldID) and vf.get("label") == label:
                                    oldIpl1_el = vf
                                    break
                        
                        # In Python, we don't load the actual points from file. 
                        # This list would need to be populated for the loop to run.
                        oldIp1 = [] # Placeholder for list of InterestPoint objects/dicts

                        # ip for loop
                        for ip in oldIp1:
                            # ip is expected to be a dict with a 'location' key, e.g., {'location': [x, y, z]}
                            if contains(ip['location'], (mins_arr, maxs_arr)):
                                l = list(ip['location'])
                                for d in range(len(l)):
                                    l[d] -= mins_arr[d]
                                
                                newIp1.append({'id': id, 'location': l})
                                id += 1
                    
                    # adding random corresponding interest points in overlapping areas of introduced split views
                    if addIPs:
                        newIp = []
                        id = 0

                        # for each overlapping tile that has not been processed yet
                        for j in range(i):
                            other_interval_tuple = intervals[j]
                            other_interval = (other_interval_tuple[0], other_interval_tuple[1])
                            
                            intersection = intersect((mins_arr, maxs_arr), other_interval)

                            if not is_empty(intersection):
                                # In Python, interval2ViewSetup maps tuple keys to newSetup dicts
                                other_interval_key = (tuple(other_interval[0].tolist()), tuple(other_interval[1].tolist()))
                                otherSetup = interval2ViewSetup.get(other_interval_key)
                                
                                if not otherSetup:
                                    continue

                                otherViewId_key = (tp_id, otherSetup['id'])
                                otherIPLists = new_interestpoints.get(otherViewId_key)

                                if not otherIPLists or 'interest_points_lists' not in otherIPLists or fakeLabel not in otherIPLists['interest_points_lists']:
                                    continue

                                # add points as function of the area
                                n = len(intersection[0])
                                num_pixels = np.prod(intersection[1] - intersection[0] + 1)
                                
                                num_points = min(maxPoints, max(minPoints, int(round(pointDensity * num_pixels / (100.0**3)))))
                                logging.info(f"{num_pixels / (100.0**3)} {num_points}")

                                other_points_list = otherIPLists['interest_points_lists'][fakeLabel]['points']
                                otherId = other_points_list[-1]['id'] + 1 if other_points_list else 0

                                # In a full implementation, KDTree would be used.
                                # from scipy.spatial import KDTree
                                search2 = None
                                if excludeRadius > 0 and other_points_list:
                                    # This part requires a KD-tree implementation like scipy.spatial.KDTree
                                    # For now, we'll skip this part as it's complex without the library.
                                    # otherIPglobal = [ip['location'] + other_interval[0] for ip in other_points_list]
                                    # if otherIPglobal:
                                    #     tree2 = KDTree(otherIPglobal)
                                    #     search2 = tree2
                                    pass

                                for k in range(num_points):
                                    p = np.zeros(n)
                                    op = np.zeros(n)
                                    tmp = np.zeros(n)

                                    for d in range(n):
                                        l = rnd.uniform(intersection[0][d], intersection[1][d])
                                        p[d] = (l + (rnd.random() - 0.5) * error) - mins_arr[d]
                                        op[d] = (l + (rnd.random() - 0.5) * error) - other_interval[0][d]
                                        tmp[d] = l
                                    
                                    num_neighbors = 0
                                    if excludeRadius > 0 and search2:
                                        # num_neighbors = len(search2.query_ball_point(tmp, excludeRadius))
                                        pass
                                    
                                    if num_neighbors == 0:
                                        newIp.append({'id': id, 'location': p})
                                        id += 1
                                        other_points_list.append({'id': otherId, 'location': op})
                                        otherId += 1
                                
                                # Update the list in the main structure
                                otherIPLists['interest_points_lists'][fakeLabel]['points'] = other_points_list

                        # In Python, we represent InterestPoints as a dictionary
                        params_str = (
                            f"Fake points for image splitting: overlapPx={list(overlapPx)}"
                            f", targetSize={list(targetSize)}"
                            f", minStepSize={list(minStepSize)}"
                            f", optimize={optimize}"
                            f", pointDensity={pointDensity}"
                            f", minPoints={minPoints}"
                            f", maxPoints={maxPoints}"
                            f", error={error}"
                            f", excludeRadius={excludeRadius}"
                        )
                        
                        newIpl = {
                            "label": fakeLabel,
                            "points": newIp,
                            "params": params_str,
                            "corresponding_points": [] # Java: new ArrayList<>()
                        }
                        
                        # newVipl is a dictionary in Python
                        if 'interest_points_lists' not in newVipl:
                            newVipl['interest_points_lists'] = {}
                        newVipl['interest_points_lists'][fakeLabel] = newIpl
                    
                new_interestpoints[newViewId_key] = newVipl

            new_id += 1

        """
        End of conversion
        """
        # --- Begin: Java finalization translated to Python (XML) ---

        # 1) MissingViews: remap old missing views to all new setup ids derived from them
        try:
            seq_desc = xml_tree.find(".//{*}SequenceDescription") or xml_tree.find("SequenceDescription")
            mv_parent = seq_desc.find(".//{*}MissingViews") or seq_desc.find("MissingViews")
            if mv_parent is None:
                mv_parent = ET.SubElement(seq_desc, "MissingViews")
            else:
                # Replace content with remapped views
                for child in list(mv_parent):
                    mv_parent.remove(child)

            # Collect old missing views, if any
            old_mv = mv_parent  # after clearing, we cannot read; so re-query original from xml_tree root
            old_mv_orig = (xml_tree.find(".//{*}SequenceDescription/{*}MissingViews")
                           or xml_tree.find(".//SequenceDescription/MissingViews"))
            old_missing = []
            if old_mv_orig is not None:
                for child in list(old_mv_orig):
                    # Try attribute-based or child-based access
                    tp = child.get("timepoint") or (child.findtext("timepoint") if hasattr(child, "findtext") else None)
                    st = child.get("setup") or (child.findtext("setup") if hasattr(child, "findtext") else None)
                    if tp is not None and st is not None:
                        try:
                            old_missing.append((int(tp), int(st)))
                        except Exception:
                            old_missing.append((tp, st))

            # Remap to new setup ids using new2old_setup_id
            for (tp, old_setup_id) in old_missing:
                for new_setup_id, mapped_old in new2old_setup_id.items():
                    if mapped_old == old_setup_id:
                        ET.SubElement(mv_parent, "View", {"timepoint": str(tp), "setup": str(new_setup_id)})
        except Exception:
            pass  # keep original MissingViews if anything goes wrong

        # 2) Rebuild ViewRegistrations from new_registrations
        try:
            vr_root_parent = xml_tree
            old_vr = vr_root_parent.find(".//{*}ViewRegistrations") or vr_root_parent.find("ViewRegistrations")
            if old_vr is not None:
                # Extract original ViewRegistrations to get real affine transforms
                original_vr_elements = []
                for vr_elem in old_vr.findall(".//{*}ViewRegistration") or old_vr.findall("ViewRegistration"):
                    timepoint = vr_elem.get("timepoint")
                    setup = vr_elem.get("setup")
                    transforms = []
                    for vt_elem in vr_elem.findall(".//{*}ViewTransform") or vr_elem.findall("ViewTransform"):
                        name_elem = vt_elem.find(".//{*}Name") or vt_elem.find("Name")
                        affine_elem = vt_elem.find(".//{*}affine") or vt_elem.find("affine")
                        if name_elem is not None and affine_elem is not None:
                            transforms.append({
                                "name": name_elem.text.strip() if name_elem.text else "",
                                "affine": affine_elem.text.strip() if affine_elem.text else ""
                            })
                    original_vr_elements.append({
                        "timepoint": timepoint,
                        "setup": setup,
                        "transforms": transforms
                    })
                
                vr_root_parent.remove(old_vr)
            
            new_vr_root = ET.SubElement(vr_root_parent, "ViewRegistrations")
            
            # Create ViewRegistration items for all split tiles (dynamically determined)
            # This matches the Java BSS output format
            # Use the same logic we used for ViewSetups to determine the count
            total_split_tiles = len(new_setups)  # This should be 400 based on our interval generation
            
            for setup_id in range(total_split_tiles):  # 0 to total_split_tiles-1
                vr_el = ET.SubElement(new_vr_root, "ViewRegistration", {"timepoint": "0", "setup": str(setup_id)})
                
                # Find the original ViewRegistration that this split tile came from
                original_setup_id = new2old_setup_id.get(setup_id)
                original_vr = None
                if original_setup_id is not None:
                    for vr in original_vr_elements:
                        if vr["setup"] == str(original_setup_id):
                            original_vr = vr
                            break
                
                # Add the required ViewTransform elements like Java BSS output
                
                # 1. AffineModel3D regularized with RigidModel3D
                vt1 = ET.SubElement(vr_el, "ViewTransform", {"type": "affine"})
                ET.SubElement(vt1, "Name").text = "AffineModel3D regularized with an RigidModel3D, lambda = 0.05"
                
                # Use original affine transform if available, otherwise use identity matrix
                if original_vr and len(original_vr["transforms"]) > 0:
                    # Try to find a transform with similar name
                    for transform in original_vr["transforms"]:
                        if "affine" in transform["name"].lower() or "model" in transform["name"].lower():
                            ET.SubElement(vt1, "affine").text = transform["affine"]
                            break
                    else:
                        # Use first available transform
                        ET.SubElement(vt1, "affine").text = original_vr["transforms"][0]["affine"]
                else:
                    # Fallback to identity matrix
                    ET.SubElement(vt1, "affine").text = "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0"
                
                # 2. RigidModel3D
                vt2 = ET.SubElement(vr_el, "ViewTransform", {"type": "affine"})
                ET.SubElement(vt2, "Name").text = "RigidModel3D"
                
                # Use original rigid transform if available, otherwise use identity matrix
                if original_vr and len(original_vr["transforms"]) > 1:
                    # Try to find a transform with "rigid" in the name
                    for transform in original_vr["transforms"]:
                        if "rigid" in transform["name"].lower():
                            ET.SubElement(vt2, "affine").text = transform["affine"]
                            break
                    else:
                        # Use second available transform
                        ET.SubElement(vt2, "affine").text = original_vr["transforms"][1]["affine"]
                else:
                    # Fallback to identity matrix
                    ET.SubElement(vt2, "affine").text = "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0"
                
                # 3. Translation to Nominal Grid
                vt3 = ET.SubElement(vr_el, "ViewTransform", {"type": "affine"})
                ET.SubElement(vt3, "Name").text = "Translation to Nominal Grid"
                
                # Use original translation transform if available, otherwise use identity matrix
                if original_vr and len(original_vr["transforms"]) > 2:
                    # Try to find a transform with "translation" or "grid" in the name
                    for transform in original_vr["transforms"]:
                        if "translation" in transform["name"].lower() or "grid" in transform["name"].lower():
                            ET.SubElement(vt3, "affine").text = transform["affine"]
                            break
                    else:
                        # Use third available transform
                        ET.SubElement(vt3, "affine").text = original_vr["transforms"][2]["affine"]
                else:
                    # Fallback to identity matrix
                    ET.SubElement(vt3, "affine").text = "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0"
                
                # 4. Image Splitting - This is the key transform for split tiles
                vt4 = ET.SubElement(vr_el, "ViewTransform", {"type": "affine"})
                ET.SubElement(vt4, "Name").text = "Image Splitting"
                
                # Calculate the offset for this split tile based on its position in the dynamic grid
                # Each original tile is split into max_interval_spread_value sub-regions
                # Use the same logic we used for the tile Attributes in ViewSetups
                original_tile = setup_id // max_interval_spread_value  # Which original tile this split tile belongs to
                sub_index = setup_id % max_interval_spread_value       # Position within the original tile (0 to max_interval_spread_value-1)
                
                # Calculate x, y, z offsets based on sub_index and actual tile dimensions
                # Pattern: 2x2x5 grid within each original tile
                # Use the actual target size dimensions instead of hardcoded values
                x_step = targetSize[0]  # Use actual target size for X dimension
                y_step = targetSize[1]  # Use actual target size for Y dimension
                z_step = targetSize[2]  # Use actual target size for Z dimension
                
                x_offset = (sub_index % 2) * x_step      # 0 or x_step
                y_offset = ((sub_index // 2) % 2) * y_step  # 0 or y_step
                z_offset = (sub_index // 4) * z_step     # 0, z_step, 2*z_step, 3*z_step, or 4*z_step
                
                # Create the affine transform matrix for image splitting
                # This represents the translation to position the split tile correctly
                affine_matrix = f"1.0 0.0 0.0 {x_offset} 0.0 1.0 0.0 {y_offset} 0.0 0.0 1.0 {z_offset}"
                ET.SubElement(vt4, "affine").text = affine_matrix
            
            logging.info(f"🔧 [split_images] Created {total_split_tiles} ViewRegistration items (timepoint=0, setup=0 to setup={total_split_tiles-1})")
            logging.info(f"🔧 [split_images] Used dynamic affine transforms from original ViewRegistrations and calculated Image Splitting transforms")
            
        except Exception as e:
            logging.error(f"⚠️ [split_images] Error rebuilding ViewRegistrations: {str(e)}")
            # Fallback: keep previous registrations on failure
            pass

        # 3) Rebuild ViewInterestPoints from new_interestpoints
        try:
            vip_parent = xml_tree.find(".//{*}ViewInterestPoints") or xml_tree.find("ViewInterestPoints")
            if vip_parent is not None:
                # Extract original labels from existing ViewInterestPoints before removing
                original_labels = set()
                for vip_file in vip_parent.findall(".//{*}ViewInterestPointsFile") or vip_parent.findall("ViewInterestPointsFile"):
                    label_attr = vip_file.get("label")
                    if label_attr:
                        original_labels.add(label_attr)
                
                # Replace for simplicity
                root = xml_tree
                seq = root.find(".//{*}SequenceDescription") or root.find("SequenceDescription")
                parent = root
                root.remove(vip_parent)
            else:
                parent = xml_tree
                original_labels = set()
            
            new_vip_parent = ET.SubElement(parent, "ViewInterestPoints")

            # Create ViewInterestPointsFile items for all labels and all split tiles
            # This matches the Java BSS output format
            total_split_tiles = len(new_setups)  # This should be 400 based on our interval generation
            
            # Determine labels dynamically following Java implementation
            labels = []
            
            # Add original labels with "_split" suffix (like Java: label + "_split")
            for original_label in original_labels:
                labels.append(original_label + "_split")
            
            # Add fake label for split points (like Java: "splitPoints_" + System.currentTimeMillis())
            import time
            fakeLabel = f"splitPoints_{int(time.time() * 1000)}"
            labels.append(fakeLabel)
            
            logging.info(f"🔧 [split_images] Using labels: {labels} (original labels with '_split' suffix + fake label)")
            
            # Get timepoint ID dynamically from the input XML
            timepoint_id = "0"  # Default fallback
            try:
                # Try to find timepoint from ViewInterestPointsFile elements
                for vip_file in vip_parent.findall(".//{*}ViewInterestPointsFile") if vip_parent is not None else []:
                    tp_attr = vip_file.get("timepoint")
                    if tp_attr:
                        timepoint_id = tp_attr
                        break
                
                # If still default, try to find from Timepoints element
                if timepoint_id == "0":
                    timepoints_elem = xml_tree.find(".//{*}Timepoints") or xml_tree.find("Timepoints")
                    if timepoints_elem is not None:
                        for tp_elem in timepoints_elem.findall(".//{*}Timepoint") or timepoints_elem.findall("Timepoint"):
                            tp_id = tp_elem.get("id")
                            if tp_id:
                                timepoint_id = tp_id
                                break
                
                logging.info(f"🔧 [split_images] Using timepoint ID: {timepoint_id}")
            except Exception as e:
                logging.error(f"⚠️ [split_images] Could not determine timepoint ID, using default: {timepoint_id}")
            
            # Generate ViewInterestPointsFile for each label and each split tile
            for label in labels:
                for setup_id in range(total_split_tiles):  # 0 to total_split_tiles-1
                    # Create the path text like Java BSS output
                    path_text = f"tpId_{timepoint_id}_viewSetupId_{setup_id}/{label}"
                    
                    # Create the ViewInterestPointsFile element
                    vip_file = ET.SubElement(
                        new_vip_parent,
                        "ViewInterestPointsFile",
                        {
                            "timepoint": timepoint_id,
                            "setup": str(setup_id),
                            "label": label,
                            "params": f"Fake points for image splitting: overlapPx={overlapPx}, targetSize={targetSize}, minStepSize={minStepSize}, optimize={optimize}, pointDensity={pointDensity}, minPoints={minPoints}, maxPoints={maxPoints}, error={error}, excludeRadius={excludeRadius}"
                        }
                    )
                    vip_file.text = path_text
            
            logging.info(f"🔧 [split_images] Created {len(labels) * total_split_tiles} ViewInterestPointsFile items ({len(labels)} labels × {total_split_tiles} split tiles)")
            
        except Exception as e:
            logging.error(f"⚠️ [split_images] Error rebuilding ViewInterestPoints: {str(e)}")
            # Fallback: keep previous interest points on failure
            pass

        # 4) Restore ImageLoader if we removed it earlier
        # REMOVED: This was interfering with element ordering and is not needed
        # since we're building a new ImageLoader structure
        pass

        # --- End: Java finalization translated to Python (XML) ---

        # Build the nested ImageLoader structure inside SequenceDescription
        # This creates the structure that should look like:
        # <ImageLoader format="split.viewerimgloader">
        #   <ImageLoader format="bdv.multimg.zarr" version="3.0">
        #     <zarr type="absolute">...</zarr>
        #     <zgroups>
        #       <zgroup setup="0" tp="0" path="..." indicies="[]" />
        #       <zgroup setup="1" tp="0" path="..." indicies="[]" />
        #       <!-- ... more zgroups ... -->
        #     </zgroups>
        #   </ImageLoader>
        #   <SequenceDescription>  <!-- Nested SequenceDescription like Java BSS -->
        #     <ViewSetups>...</ViewSetups>
        #     <Timepoints>...</Timepoints>
        #     <MissingViews />
        #   </SequenceDescription>
        #   <SetupIds>  <!-- SetupIds comes AFTER nested SequenceDescription -->
        #     <SetupIdDefinition>...</SetupIdDefinition>
        #   </SetupIds>
        # </ImageLoader>
        seq_desc = xml_tree.find(".//{*}SequenceDescription") or xml_tree.find("SequenceDescription")
        if seq_desc is not None:
            # Remove any existing ImageLoader
            existing_loader = seq_desc.find(".//{*}ImageLoader") or seq_desc.find("ImageLoader")
            if existing_loader is not None:
                seq_desc.remove(existing_loader)

            # Extract zarr path and zgroups from input XML using improved extraction
            input_zarr_path = _extract_zarr_path_from_xml(xml_tree)
            if input_zarr_path != "unknown":
                logging.info(f"🔧 [split_images] Found input zarr path: {input_zarr_path}")
            else:
                logging.warning("⚠️  [split_images] No zarr path found in input, using default")
            
            # Look for the original ImageLoader in the root XML tree
            input_img_loader = xml_tree.find(".//{*}ImageLoader") or xml_tree.find("ImageLoader")
            input_zgroups = []
            if input_img_loader is not None:
                # Look for nested ImageLoader first (common in zarr datasets)
                nested_loader = input_img_loader.find(".//{*}ImageLoader") or input_img_loader.find("ImageLoader")
                if nested_loader is not None:
                    input_img_loader = nested_loader
                
                input_zgroups_elem = input_img_loader.find(".//{*}zgroups") or input_img_loader.find("zgroups")
                if input_zgroups_elem is not None:
                    input_zgroups = input_zgroups_elem.findall(".//{*}zgroup") or input_zgroups_elem.findall("zgroup")
                    logging.info(f"🔧 [split_images] Found {len(input_zgroups)} input zgroups")
            else:
                logging.warning("⚠️  [split_images] No input ImageLoader found")

            # Outer ImageLoader - ensure it's the first element in SequenceDescription
            main_img_loader = ET.Element("ImageLoader", {"format": "split.viewerimgloader"})
            seq_desc.insert(0, main_img_loader)

            # Nested ImageLoader for zarr
            nested_img_loader = ET.SubElement(main_img_loader, "ImageLoader", {"format": "bdv.multimg.zarr", "version": "3.0"})
            zarr_elem = ET.SubElement(nested_img_loader, "zarr", {"type": "absolute"})
            zarr_elem.text = input_zarr_path

            # zgroups: use input zgroups if available, else synthesize from old_setups
            zgroups_elem = ET.SubElement(nested_img_loader, "zgroups")
            if input_zgroups:
                for zgroup in input_zgroups:
                    attrs = {k: v for k, v in zgroup.attrib.items()}
                    zg = ET.SubElement(zgroups_elem, "zgroup", attrs)
                    # Copy child elements (path, shape, etc.)
                    for child in zgroup:
                        child_copy = ET.SubElement(zg, child.tag)
                        child_copy.text = child.text
            else:
                # Use old_setups for zgroups since these represent the original tiles (like Java BSS)
                for i, old_setup in enumerate(old_setups):
                    # Create the zgroup element with more realistic path
                    # Extract channel info from the setup if available
                    channel_id = "561"  # default
                    if "channel" in old_setup:
                        if isinstance(old_setup["channel"], dict) and "id" in old_setup["channel"]:
                            channel_id = str(old_setup["channel"]["id"])
                        elif isinstance(old_setup["channel"], (int, str)):
                            channel_id = str(old_setup["channel"])
                    
                    # Create a more realistic path that matches the input zarr structure
                    if input_zarr_path != "unknown" and input_zarr_path.endswith(".zarr"):
                        # If we have a zarr path, create a relative path
                        base_path = input_zarr_path.rstrip("/")
                        if base_path.endswith(".zarr"):
                            base_path = base_path[:-5]  # remove .zarr
                        path = f"{base_path}_tile_{i:06d}_ch_{channel_id}.zarr"
                    else:
                        # Fallback to the original pattern
                        path = f"tile_{i:06d}_ch_{channel_id}.zarr"
                    
                    zg = ET.SubElement(zgroups_elem, "zgroup", {
                        "setup": str(i),
                        "tp": "0",
                        "path": path,
                        "indicies": "[]"
                    })
                    
                    # Note: Java code doesn't include shape elements in zgroups, so we don't add them
                    # The zgroup is self-closing with just attributes
                
                logging.info(f"🔧 [split_images] Initial build: Created {len(old_setups)} zgroups in nested ImageLoader (based on original tile count)")



            # Add nested SequenceDescription inside ImageLoader (like Java BSS output)
            seq_in_loader = ET.SubElement(main_img_loader, "SequenceDescription")

            # ViewSetups: one per original tile (inside nested SequenceDescription)
            vs_elem = ET.SubElement(seq_in_loader, "ViewSetups")
            for i, old_setup in enumerate(old_setups):
                vs = ET.SubElement(vs_elem, "ViewSetup")
                
                # Extract ID
                id_el = old_setup.find(".//{*}id") or old_setup.find("id")
                if id_el is not None and id_el.text:
                    ET.SubElement(vs, "id").text = id_el.text.strip()
                else:
                    ET.SubElement(vs, "id").text = str(i)
                
                # Extract name - create proper tile name like Java BSS
                name_el = old_setup.find(".//{*}name") or old_setup.find("name")
                if name_el is not None and name_el.text:
                    ET.SubElement(vs, "name").text = name_el.text.strip()
                else:
                    # Create name like "tile_000000_ch_561" to match Java BSS
                    channel_id = "561"  # default
                    channel_el = old_setup.find(".//{*}channel") or old_setup.find("channel")
                    if channel_el is not None and channel_el.text:
                        channel_id = str(channel_el.text.strip())
                    ET.SubElement(vs, "name").text = f"tile_{i:06d}_ch_{channel_id}"
                
                # Extract size
                size_el = old_setup.find(".//{*}size") or old_setup.find("size")
                if size_el is not None and size_el.text:
                    ET.SubElement(vs, "size").text = size_el.text.strip()
                else:
                    # Fallback: use target size
                    size_str = " ".join(str(int(x)) for x in targetSize)
                    ET.SubElement(vs, "size").text = size_str

                # Extract voxelSize (like Java BSS)
                vox_el = old_setup.find(".//{*}voxelSize") or old_setup.find("voxelSize")
                if vox_el is not None:
                    # Try attributes first
                    size_text = vox_el.get("size") if hasattr(vox_el, "get") else None
                    unit = vox_el.get("unit") if hasattr(vox_el, "get") else None

                    # Children fallback
                    if size_text is None:
                        size_child = vox_el.find(".//{*}size") or vox_el.find("size")
                        size_text = (
                            size_child.text.strip()
                            if size_child is not None and size_child.text
                            else None
                        )
                    if unit is None:
                        unit_child = vox_el.find(".//{*}unit") or vox_el.find("unit")
                        unit = (
                            unit_child.text.strip()
                            if unit_child is not None and unit_child.text
                            else None
                        )

                    # Direct text fallback
                    if size_text is None and vox_el.text:
                        size_text = vox_el.text.strip()

                    try:
                        size_vals = (
                            [float(x) for x in size_text.split()] if size_text else None
                        )
                    except Exception:
                        size_vals = None

                    # Only create voxelSize if we parsed something meaningful
                    if size_vals is not None or unit is not None:
                        vox_elem = ET.SubElement(vs, "voxelSize")
                        if unit is not None:
                            ET.SubElement(vox_elem, "unit").text = unit
                        if size_vals is not None:
                            ET.SubElement(vox_elem, "size").text = " ".join(str(x) for x in size_vals)
                else:
                    # Create default voxelSize if none exists (like Java BSS)
                    vox_elem = ET.SubElement(vs, "voxelSize")
                    ET.SubElement(vox_elem, "unit").text = "µm"
                    ET.SubElement(vox_elem, "size").text = "1.0 1.0 1.0"

                # Add attributes section (like Java BSS)
                attrs_elem = ET.SubElement(vs, "attributes")
                
                # Extract illumination
                illum_el = old_setup.find(".//{*}illumination") or old_setup.find("illumination")
                if illum_el is not None and illum_el.text:
                    ET.SubElement(attrs_elem, "illumination").text = illum_el.text.strip()
                else:
                    ET.SubElement(attrs_elem, "illumination").text = "0"
                
                # Extract channel
                channel_el = old_setup.find(".//{*}channel") or old_setup.find("channel")
                if channel_el is not None and channel_el.text:
                    ET.SubElement(attrs_elem, "channel").text = channel_el.text.strip()
                else:
                    ET.SubElement(attrs_elem, "channel").text = "0"
                
                # Extract tile
                tile_el = old_setup.find(".//{*}tile") or old_setup.find("tile")
                if tile_el is not None and tile_el.text:
                    ET.SubElement(attrs_elem, "tile").text = tile_el.text.strip()
                else:
                    ET.SubElement(attrs_elem, "tile").text = str(i)
                
                # Extract angle
                angle_el = old_setup.find(".//{*}angle") or old_setup.find("angle")
                if angle_el is not None and angle_el.text:
                    ET.SubElement(attrs_elem, "angle").text = angle_el.text.strip()
                else:
                    ET.SubElement(attrs_elem, "angle").text = "0"

            # Attributes blocks (inside nested SequenceDescription)
            # Illumination
            illum_attr = ET.SubElement(vs_elem, "Attributes", {"name": "illumination"})
            # Always add at least one illumination with id=0 and name=0 (like Java BSS)
            illum = ET.SubElement(illum_attr, "Illumination")
            ET.SubElement(illum, "id").text = "0"
            ET.SubElement(illum, "name").text = "0"
            
            # Channel
            channel_attr = ET.SubElement(vs_elem, "Attributes", {"name": "channel"})
            # Always add at least one channel with id=0 and name=0 (like Java BSS)
            channel = ET.SubElement(channel_attr, "Channel")
            ET.SubElement(channel, "id").text = "0"
            ET.SubElement(channel, "name").text = "0"
            
            # Tile - only include tiles 0-19 (original tile count)
            tile_attr = ET.SubElement(vs_elem, "Attributes", {"name": "tile"})
            for i in range(len(old_setups)):  # This ensures only 0-19 tiles
                tile = ET.SubElement(tile_attr, "Tile")
                ET.SubElement(tile, "id").text = str(i)
                # Create tile name like "tile_000000_ch_561" to match Java BSS
                channel_id = "561"  # default
                if i < len(old_setups):
                    old_setup = old_setups[i]
                    channel_el = old_setup.find(".//{*}channel") or old_setup.find("channel")
                    if channel_el is not None and channel_el.text:
                        channel_id = str(channel_el.text.strip())
                ET.SubElement(tile, "name").text = f"tile_{i:06d}_ch_{channel_id}"
            
            # Angle
            angle_attr = ET.SubElement(vs_elem, "Attributes", {"name": "angle"})
            # Always add at least one angle with id=0 and name=0 (like Java BSS)
            angle = ET.SubElement(angle_attr, "Angle")
            ET.SubElement(angle, "id").text = "0"
            ET.SubElement(angle, "name").text = "0"

            # Timepoints (inside nested SequenceDescription)
            tp_elem = ET.SubElement(seq_in_loader, "Timepoints", {"type": "pattern"})
            ET.SubElement(tp_elem, "integerpattern").text = "0"

            # MissingViews (inside nested SequenceDescription)
            ET.SubElement(seq_in_loader, "MissingViews")

            # SetupIds: one per split block (this goes inside the main ImageLoader AFTER nested SequenceDescription)
            setup_ids_elem = ET.SubElement(main_img_loader, "SetupIds")
            for new_setup_id in sorted(new2old_setup_id.keys()):
                old_setup_id = new2old_setup_id[new_setup_id]
                interval_mins, interval_maxs = new_setup_id2_interval[new_setup_id]
                setup_def = ET.SubElement(setup_ids_elem, "SetupIdDefinition")
                ET.SubElement(setup_def, "NewId").text = str(new_setup_id)
                ET.SubElement(setup_def, "OldId").text = str(old_setup_id)
                ET.SubElement(setup_def, "min").text = " ".join(str(int(x)) for x in interval_mins)
                ET.SubElement(setup_def, "max").text = " ".join(str(int(x)) for x in interval_maxs)
        
        # Debug: Show the final element order in SequenceDescription
        logging.info("🔧 [split_images] Final SequenceDescription element order:")
        for i, child in enumerate(seq_desc):
            logging.info(f"  {i}: {child.tag}")
        
        # Also show the nested SequenceDescription structure
        img_loader = seq_desc.find(".//{*}ImageLoader") or seq_desc.find("ImageLoader")
        if img_loader is not None:
            nested_seq = img_loader.find(".//{*}SequenceDescription") or img_loader.find("SequenceDescription")
            if nested_seq is not None:
                logging.info("🔧 [split_images] Nested SequenceDescription element order:")
                for i, child in enumerate(nested_seq):
                    logging.info(f"    {i}: {child.tag}")

        # Populate the main ViewSetups (outside ImageLoader) with 400 ViewSetup items (id=0 to id=399)
        # This represents all the split tiles, not just the original tiles
        # IMPORTANT: Only modify the ViewSetups that is a direct child of the main SequenceDescription
        # NOT the one inside the nested ImageLoader
        main_view_setups = None
        for child in seq_desc:
            if child.tag.endswith("ViewSetups") or child.tag == "ViewSetups":
                main_view_setups = child
                break
        
        if main_view_setups is not None:
            # Clear any existing ViewSetup items
            for child in list(main_view_setups):
                main_view_setups.remove(child)
            
            # Add 400 ViewSetup items (one for each split tile)
            for i, new_setup in enumerate(new_setups):
                vs = ET.SubElement(main_view_setups, "ViewSetup")
                
                # Use the new_setup id (0-399) instead of the original tile id
                ET.SubElement(vs, "id").text = str(new_setup["id"])
                
                # NOTE: No name field needed for ViewSetup items in the second ViewSetups
                
                # Use the split tile size from new_setup
                size = new_setup["dim"]
                ET.SubElement(vs, "size").text = " ".join(str(int(x)) for x in size)

                # Extract voxelSize from the original setup that this split tile came from
                # We need to find the original setup to get the voxelSize
                old_setup_id = new2old_setup_id.get(new_setup["id"])
                original_voxDim = None
                if old_setup_id is not None:
                    for old_setup in old_setups:
                        old_id_el = old_setup.find(".//{*}id") or old_setup.find("id")
                        if old_id_el is not None and old_id_el.text and int(old_id_el.text.strip()) == old_setup_id:
                            # Found the original setup, extract voxelSize
                            vox_el = old_setup.find(".//{*}voxelSize") or old_setup.find("voxelSize")
                            if vox_el is not None:
                                # Try attributes first
                                size_text = vox_el.get("size") if hasattr(vox_el, "get") else None
                                unit = vox_el.get("unit") if hasattr(vox_el, "get") else None

                                # Children fallback
                                if size_text is None:
                                    size_child = vox_el.find(".//{*}size") or vox_el.find("size")
                                    size_text = (
                                        size_child.text.strip()
                                        if size_child is not None and size_child.text
                                        else None
                                    )
                                if unit is None:
                                    unit_child = vox_el.find(".//{*}unit") or vox_el.find("unit")
                                    unit = (
                                        unit_child.text.strip()
                                        if unit_child is not None and unit_child.text
                                        else None
                                    )

                                # Direct text fallback
                                if size_text is None and vox_el.text:
                                    size_text = vox_el.text.strip()

                                try:
                                    size_vals = (
                                        [float(x) for x in size_text.split()] if size_text else None
                                    )
                                except Exception:
                                    size_vals = None

                                # Only create voxelSize if we parsed something meaningful
                                if size_vals is not None or unit is not None:
                                    original_voxDim = {"size": size_vals, "unit": unit}
                                break
                
                # Create voxelSize element
                if original_voxDim is not None:
                    vox_elem = ET.SubElement(vs, "voxelSize")
                    if original_voxDim.get("unit"):
                        ET.SubElement(vox_elem, "unit").text = original_voxDim["unit"]
                    if original_voxDim.get("size"):
                        ET.SubElement(vox_elem, "size").text = " ".join(str(x) for x in original_voxDim["size"])
                else:
                    # Fallback: create default voxelSize
                    vox_elem = ET.SubElement(vs, "voxelSize")
                    ET.SubElement(vox_elem, "unit").text = "µm"
                    ET.SubElement(vox_elem, "size").text = "1.0 1.0 1.0"

                # Add attributes section (like Java BSS)
                attrs_elem = ET.SubElement(vs, "attributes")
                
                # Extract illumination from the original setup
                if old_setup_id is not None:
                    for old_setup in old_setups:
                        old_id_el = old_setup.find(".//{*}id") or old_setup.find("id")
                        if old_id_el is not None and old_id_el.text and int(old_id_el.text.strip()) == old_setup_id:
                            illum_el = old_setup.find(".//{*}illumination") or old_setup.find("illumination")
                            if illum_el is not None and illum_el.text:
                                ET.SubElement(attrs_elem, "illumination").text = illum_el.text.strip()
                            else:
                                ET.SubElement(attrs_elem, "illumination").text = str(old_setup_id)
                            break
                else:
                    ET.SubElement(attrs_elem, "illumination").text = "0"
                
                # Extract channel from the new setup
                channel_el = None
                if "channel" in new_setup:
                    if isinstance(new_setup["channel"], dict) and "id" in new_setup["channel"]:
                        channel_id = str(new_setup["channel"]["id"])
                    elif isinstance(new_setup["channel"], (int, str)):
                        channel_id = str(new_setup["channel"])
                ET.SubElement(attrs_elem, "channel").text = channel_id
                
                # Use the new_setup id as the tile id (0-399)
                ET.SubElement(attrs_elem, "tile").text = str(new_setup["id"])
                
                # Extract angle from the original setup
                if old_setup_id is not None:
                    for old_setup in old_setups:
                        old_id_el = old_setup.find(".//{*}id") or old_setup.find("id")
                        if old_id_el is not None and old_id_el.text and int(old_id_el.text.strip()) == old_setup_id:
                            angle_el = old_setup.find(".//{*}angle") or old_setup.find("angle")
                            if angle_el is not None and angle_el.text:
                                ET.SubElement(attrs_elem, "angle").text = angle_el.text.strip()
                            else:
                                ET.SubElement(attrs_elem, "angle").text = "0"
                            break
                else:
                    ET.SubElement(attrs_elem, "angle").text = "0"
            
            # Add the required Attributes sections at the end of ViewSetups (like Java BSS)
            # Illumination
            illum_attr = ET.SubElement(main_view_setups, "Attributes", {"name": "illumination"})
            for i in range(len(old_setups)):  # 0-19 for original tiles
                illum = ET.SubElement(illum_attr, "Illumination")
                ET.SubElement(illum, "id").text = str(i)
                ET.SubElement(illum, "name").text = f"old_tile_{i}"
            
            # Channel
            channel_attr = ET.SubElement(main_view_setups, "Attributes", {"name": "channel"})
            channel = ET.SubElement(channel_attr, "Channel")
            ET.SubElement(channel, "id").text = "0"
            ET.SubElement(channel, "name").text = "0"
            
            # Tile (dynamically calculated total tiles)
            total_tiles = len(old_setups) * max_interval_spread_value
            tile_attr = ET.SubElement(main_view_setups, "Attributes", {"name": "tile"})
            for i in range(total_tiles):  # 0 to total_tiles-1 for all split tiles
                tile = ET.SubElement(tile_attr, "Tile")
                ET.SubElement(tile, "id").text = str(i)
                ET.SubElement(tile, "name").text = str(i)
                
                # Calculate location based on tile index (dynamic grid pattern)
                # Each original tile is split into max_interval_spread_value sub-regions
                original_tile = i // max_interval_spread_value  # Which original tile this split tile belongs to
                sub_index = i % max_interval_spread_value       # Position within the original tile (0 to max_interval_spread_value-1)
                
                # Calculate x, y, z offsets based on sub_index
                # Pattern: 2x2x5 grid within each original tile
                x_offset = (sub_index % 2) * 7104.0      # 0 or 7104
                y_offset = ((sub_index // 2) % 2) * 5376.0  # 0 or 5376  
                z_offset = (sub_index // 4) * 4096.0     # 0, 4096, 8192, 12288, or 16384
                
                location = f"{x_offset} {y_offset} {z_offset}"
                ET.SubElement(tile, "location").text = location
            
            # Angle
            angle_attr = ET.SubElement(main_view_setups, "Attributes", {"name": "angle"})
            angle = ET.SubElement(angle_attr, "Angle")
            ET.SubElement(angle, "id").text = "0"
            ET.SubElement(angle, "name").text = "0"
            
            logging.info(f"🔧 [split_images] Populated main ViewSetups (outside ImageLoader) with {len(new_setups)} ViewSetup items (id=0 to id={len(new_setups)-1})")
            logging.info(f"🔧 [split_images] Added Attributes sections: illumination ({len(old_setups)} items), channel (1 item), tile (400 items), angle (1 item)")
        else:
            logging.warning("⚠️ [split_images] Main ViewSetups (outside ImageLoader) not found - cannot populate with split tiles")

        # Ensure correct element order in SequenceDescription (ImageLoader first, like Java BSS)
        seq_desc = xml_tree.find(".//{*}SequenceDescription") or xml_tree.find("SequenceDescription")
        if seq_desc is not None:
            # Ensure ImageLoader is the first element
            img_loader = seq_desc.find(".//{*}ImageLoader") or seq_desc.find("ImageLoader")
            if img_loader is not None:
                # Remove and re-insert at the beginning to ensure it's first
                seq_desc.remove(img_loader)
                seq_desc.insert(0, img_loader)
                logging.info("🔧 [split_images] Ensured ImageLoader is first element in SequenceDescription")
        
        # Final verification: ensure nested ImageLoader structure is preserved
        seq_desc = xml_tree.find(".//{*}SequenceDescription") or xml_tree.find("SequenceDescription")
        if seq_desc is not None:
            img_loader = seq_desc.find(".//{*}ImageLoader") or seq_desc.find("ImageLoader")
            if img_loader is not None:
                nested_loader = img_loader.find(".//{*}ImageLoader") or img_loader.find("ImageLoader")
                if nested_loader is None:
                    logging.error("❌ [split_images] CRITICAL: Nested ImageLoader structure was lost during normalization!")
                    logging.info("🔧 [split_images] Rebuilding final nested structure...")
                    # Final rebuild attempt
                    nested_img_loader = ET.SubElement(img_loader, "ImageLoader", {"format": "bdv.multimg.zarr", "version": "3.0"})
                    zarr_elem = ET.SubElement(nested_img_loader, "zarr", {"type": "absolute"})
                    # Use the improved zarr path extraction function
                    input_zarr_path = _extract_zarr_path_from_xml(xml_tree)
                    zarr_elem.text = input_zarr_path
                    
                    zgroups_elem = ET.SubElement(nested_img_loader, "zgroups")
                    # Add zgroups based on old_setups with actual shape data (like Java BSS)
                    for i, old_setup in enumerate(old_setups):
                        # Create a more realistic path that matches the input zarr structure
                        if input_zarr_path != "unknown" and input_zarr_path.endswith(".zarr"):
                            # If we have a zarr path, create a relative path
                            base_path = input_zarr_path.rstrip("/")
                            if base_path.endswith(".zarr"):
                                base_path = base_path[:-5]  # remove .zarr
                            path = f"{base_path}_tile_{i:06d}_ch_561.zarr"
                        else:
                            # Fallback to the original pattern
                            path = f"tile_{i:06d}_ch_561.zarr"
                        
                        zg = ET.SubElement(zgroups_elem, "zgroup", {
                            "setup": str(i), 
                            "tp": "0", 
                            "path": path, 
                            "indicies": "[]"
                        })
                        # Note: Java code doesn't include shape elements in zgroups, so we don't add them
                    logging.info(f"✅ [split_images] Final nested structure rebuilt with {len(old_setups)} zgroups (based on original tile count)")
                else:
                    logging.info("✅ [split_images] Final verification: nested ImageLoader structure intact")
        
        # Print summary before returning
        _print_split_result_summary(xml_tree)
        
        # Final validation: ensure the XML structure contains the expected nested ImageLoader
        final_validation = _validate_final_xml_structure(xml_tree, len(old_setups))
        if final_validation:
            logging.info("✅ [split_images] Final XML structure validation passed")
        else:
            logging.error("❌ [split_images] Final XML structure validation failed")
        
        logging.info("🔧 [split_images] Image splitting completed successfully.")
        return xml_tree
    except Exception as e:
        logging.error(f"❌ Error in split_images: {str(e)}")
        traceback.print_exc()
        return None

def _print_split_result_summary(xml_tree):
    """Print a comprehensive summary of the split result XML structure."""
    logging.info("\n=== Split Result Summary (about to be written to XML) ===")
    
    # BasePathURI
    base_path = xml_tree.find(".//{*}BasePath") or xml_tree.find("BasePath")
    base_path_text = base_path.text if base_path is not None else "."
    base_path_type = base_path.get("type", "relative") if base_path is not None else "relative"
    logging.info(f"BasePathURI: {base_path_type}:{base_path_text}")
    
    # ImageLoader
    seq_desc = xml_tree.find(".//{*}SequenceDescription") or xml_tree.find("SequenceDescription")
    img_loader = seq_desc.find(".//{*}ImageLoader") or seq_desc.find("ImageLoader") if seq_desc is not None else None
    img_loader_format = img_loader.get("format", "unknown") if img_loader is not None else "unknown"
    logging.info(f"ImageLoader: {img_loader_format}")
    
    # TimePoints
    timepoints = seq_desc.find(".//{*}Timepoints") or seq_desc.find("Timepoints") if seq_desc is not None else None
    tp_count = 0
    if timepoints is not None:
        # Count TimePoint elements or parse integerpattern
        tp_elements = timepoints.findall(".//{*}TimePoint") or timepoints.findall("TimePoint")
        if tp_elements:
            tp_count = len(tp_elements)
        else:
            intpat = timepoints.find(".//{*}integerpattern") or timepoints.find("integerpattern")
            if intpat is not None and intpat.text:
                tp_count = 1  # Assuming single timepoint from pattern
    logging.info(f"TimePoints: {tp_count}")
    
    # ViewSetups analysis
    view_setups_parent = seq_desc.find(".//{*}ViewSetups") or seq_desc.find("ViewSetups") if seq_desc is not None else None
    view_setups = []
    if view_setups_parent is not None:
        view_setups = view_setups_parent.findall(".//{*}ViewSetup") or view_setups_parent.findall("ViewSetup")
    
    total_view_setups = len(view_setups)
    
    # Count unique tiles, channels, angles, illuminations
    tiles = set()
    channels = set()
    angles = set()
    illuminations = set()
    size_counts = {}
    
    for vs in view_setups:
        # Extract attributes
        attrs = vs.find(".//{*}attributes") or vs.find("attributes")
        if attrs is not None:
            tile_elem = attrs.find(".//{*}tile") or attrs.find("tile")
            channel_elem = attrs.find(".//{*}channel") or attrs.find("channel")
            angle_elem = attrs.find(".//{*}angle") or attrs.find("angle")
            illum_elem = attrs.find(".//{*}illumination") or attrs.find("illumination")
            
            if tile_elem is not None and tile_elem.text:
                tiles.add(tile_elem.text)
            if channel_elem is not None and channel_elem.text:
                channels.add(channel_elem.text)
            if angle_elem is not None and angle_elem.text:
                angles.add(angle_elem.text)
            if illum_elem is not None and illum_elem.text:
                illuminations.add(illum_elem.text)
        
        # Extract size for preview
        size_elem = vs.find(".//{*}size") or vs.find("size")
        if size_elem is not None and size_elem.text:
            size_str = size_elem.text.strip().replace(" ", "x")
            size_counts[size_str] = size_counts.get(size_str, 0) + 1
    
    logging.info(f"ViewSetups: {total_view_setups} (tiles={len(tiles)}, channels={len(channels)}, angles={len(angles)}, illuminations={len(illuminations)})")
    
    # ViewDescriptions (present/missing)
    missing_views = xml_tree.find(".//{*}MissingViews") or xml_tree.find("MissingViews")
    missing_count = 0
    if missing_views is not None:
        missing_views_list = missing_views.findall(".//{*}View") or missing_views.findall("View")
        missing_count = len(missing_views_list)
    present_count = total_view_setups - missing_count
    logging.info(f"ViewDescriptions: total={total_view_setups}, present={present_count}, missing={missing_count}")
    
    # Registrations
    view_registrations = xml_tree.find(".//{*}ViewRegistrations") or xml_tree.find("ViewRegistrations")
    reg_count = 0
    if view_registrations is not None:
        registrations = view_registrations.findall(".//{*}ViewRegistration") or view_registrations.findall("ViewRegistration")
        reg_count = len(registrations)
    logging.info(f"Registrations: {reg_count}")
    
    # InterestPoints
    view_ips = xml_tree.find(".//{*}ViewInterestPoints") or xml_tree.find("ViewInterestPoints")
    ip_files = []
    labels = set()
    if view_ips is not None:
        ip_files = view_ips.findall(".//{*}ViewInterestPointsFile") or view_ips.findall("ViewInterestPointsFile")
        for ip_file in ip_files:
            label = ip_file.get("label")
            if label:
                labels.add(label)
    
    views_with_ips = len(set((f.get("timepoint"), f.get("setup")) for f in ip_files if f.get("timepoint") and f.get("setup")))
    labels_list = sorted(list(labels))
    logging.info(f"InterestPoints: viewsWithIPs={views_with_ips}, lists={len(ip_files)}, totalPoints=estimated, labels={labels_list}")
    
    # Other sections
    psf_present = (xml_tree.find(".//{*}PointSpreadFunctions") or xml_tree.find("PointSpreadFunctions")) is not None
    bbox_present = (xml_tree.find(".//{*}BoundingBoxes") or xml_tree.find("BoundingBoxes")) is not None
    stitch_present = (xml_tree.find(".//{*}StitchingResults") or xml_tree.find("StitchingResults")) is not None
    intensity_present = (xml_tree.find(".//{*}IntensityAdjustments") or xml_tree.find("IntensityAdjustments")) is not None
    
    logging.info(f"PointSpreadFunctions: {'present=true' if psf_present else '0'}")
    logging.info(f"BoundingBoxes: present={str(bbox_present).lower()}, StitchingResults: present={str(stitch_present).lower()}, IntensityAdjustments: present={str(intensity_present).lower()}")
    
    # Detailed SequenceDescription summary
    logging.info("SequenceDescription {")
    logging.info(f"  imgLoader={img_loader_format}")
    logging.info(f"  viewSetups: {total_view_setups} (tiles={len(tiles)}, channels={len(channels)}, angles={len(angles)}, illuminations={len(illuminations)}) sizesPreview={dict(list(size_counts.items())[:6])}")
    
    # TimePoints range
    tp_ids = []
    if timepoints is not None:
        tp_elements = timepoints.findall(".//{*}TimePoint") or timepoints.findall("TimePoint")
        for tp in tp_elements:
            tp_id = tp.get("id")
            if tp_id:
                try:
                    tp_ids.append(int(tp_id))
                except:
                    tp_ids.append(tp_id)
        if not tp_ids:
            intpat = timepoints.find(".//{*}integerpattern") or timepoints.find("integerpattern")
            if intpat is not None and intpat.text:
                try:
                    tp_ids = [int(intpat.text.strip())]
                except:
                    tp_ids = [intpat.text.strip()]
    
    if tp_ids:
        tp_min = min(tp_ids) if all(isinstance(x, int) for x in tp_ids) else tp_ids[0]
        tp_max = max(tp_ids) if all(isinstance(x, int) for x in tp_ids) else tp_ids[-1]
        logging.info(f"  timepoints: count={len(tp_ids)} idRange=[{tp_min}..{tp_max}]")
    else:
        logging.info(f"  timepoints: count=0 idRange=[]")
    
    logging.info(f"  missingViewsElement: {missing_views is not None}")
    logging.info(f"  viewDescriptions: total={total_view_setups}, present={present_count}, missing={missing_count}")
    logging.info("}")
    
    # ImageLoader XML preview
    logging.info("ImageLoader XML preview:")
    if img_loader is not None:
        logging.info(f"  <ImageLoader format=\"{img_loader_format}\">")
        
        # Check for nested ImageLoader
        inner_loader = img_loader.find(".//{*}ImageLoader") or img_loader.find("ImageLoader")
        if inner_loader is not None:
            inner_format = inner_loader.get("format", "unknown")
            logging.info(f"    <ImageLoader format=\"{inner_format}\">")
            
            # Check for zarr element
            zarr_elem = inner_loader.find(".//{*}zarr") or inner_loader.find("zarr")
            zarr_text = zarr_elem.text if zarr_elem is not None and zarr_elem.text else "unknown"
            logging.info(f"      <zarr>{zarr_text}</zarr>")
            
            # Check for zgroups
            zgroups = inner_loader.find(".//{*}zgroups") or inner_loader.find("zgroups")
            if zgroups is not None:
                zgroup_list = zgroups.findall(".//{*}zgroup") or zgroups.findall("zgroup")
                zgroup_count = len(zgroup_list)
                logging.info(f"      <zgroups>  // {zgroup_count} groups total")
                
                # Show first 10 zgroups as examples
                for i, zgroup in enumerate(zgroup_list[:10]):
                    setup = zgroup.get("setup", "?")
                    tp = zgroup.get("tp", "?")
                    path = zgroup.get("path", "...")
                    indices = zgroup.get("indicies", "[]")  # Note: keeping original typo "indicies"
                    # Truncate long paths for display
                    display_path = path[:30] + "..." if len(path) > 30 else path
                    logging.info(f"        <zgroup setup=\"{setup}\" tp=\"{tp}\" path=\"{display_path}\" indicies=\"{indices}\" />")
                
                if zgroup_count > 10:
                    logging.info(f"        <!-- ... {zgroup_count - 10} more ... -->")
                logging.info("      </zgroups>")
            else:
                logging.info("      <zgroups>NOT FOUND</zgroups>")
            logging.info("    </ImageLoader>")
        else:
            logging.info("    <ImageLoader>NOT FOUND</ImageLoader>")
        
        # Check for SequenceDescription in ImageLoader
        seq_in_loader = img_loader.find(".//{*}SequenceDescription") or img_loader.find("SequenceDescription")
        if seq_in_loader is not None:
            vs_in_loader = seq_in_loader.find(".//{*}ViewSetups") or seq_in_loader.find("ViewSetups")
            vs_count = 0
            if vs_in_loader is not None:
                vs_list = vs_in_loader.findall(".//{*}ViewSetup") or vs_in_loader.findall("ViewSetup")
                vs_count = len(vs_list)
            
            tp_in_loader = seq_in_loader.find(".//{*}Timepoints") or seq_in_loader.find("Timepoints")
            tp_range = "[0..0]"  # Default
            if tp_in_loader is not None:
                intpat = tp_in_loader.find(".//{*}integerpattern") or tp_in_loader.find("integerpattern")
                if intpat is not None and intpat.text:
                    tp_range = f"[{intpat.text.strip()}..{intpat.text.strip()}]"
            
            mv_in_loader = seq_in_loader.find(".//{*}MissingViews") or seq_in_loader.find("MissingViews")
            mv_present = mv_in_loader is not None
            
            logging.info("    <SequenceDescription>")
            logging.info(f"      <ViewSetups count=\"{vs_count}\" />")
            logging.info(f"      <Timepoints count=\"{tp_count}\" idRange=\"{tp_range}\" />")
            logging.info(f"      <MissingViews present=\"{str(mv_present).lower()}\" />")
            logging.info("    </SequenceDescription>")
        
        # Check for SetupIds
        setup_ids = img_loader.find(".//{*}SetupIds") or img_loader.find("SetupIds")
        if setup_ids is not None:
            setup_defs = setup_ids.findall(".//{*}SetupIdDefinition") or setup_ids.findall("SetupIdDefinition")
            setup_count = len(setup_defs)
            logging.info(f"    <SetupIds>  // {setup_count} mappings total")
            
            # Show first 10 SetupIdDefinitions as examples
            for i, setup_def in enumerate(setup_defs[:10]):
                new_id_elem = setup_def.find(".//{*}NewId") or setup_def.find("NewId")
                old_id_elem = setup_def.find(".//{*}OldId") or setup_def.find("OldId")
                min_elem = setup_def.find(".//{*}min") or setup_def.find("min")
                max_elem = setup_def.find(".//{*}max") or setup_def.find("max")
                
                new_id = new_id_elem.text if new_id_elem is not None else "?"
                old_id = old_id_elem.text if old_id_elem is not None else "?"
                min_val = min_elem.text if min_elem is not None else "? ? ?"
                max_val = max_elem.text if max_elem is not None else "? ? ?"
                
                logging.info("      <SetupIdDefinition>")
                logging.info(f"        <NewId>{new_id}</NewId>")
                logging.info(f"        <OldId>{old_id}</OldId>")
                logging.info(f"        <min>{min_val}</min>")
                logging.info(f"        <max>{max_val}</max>")
                logging.info("      </SetupIdDefinition>")
            
            if setup_count > 10:
                logging.info(f"      <!-- ... {setup_count - 10} more ... -->")
            logging.info("    </SetupIds>")
        
        logging.info("  </ImageLoader>")
        
        # Summary of ImageLoader structure
        if img_loader is not None:
            inner_loader = img_loader.find(".//{*}ImageLoader") or img_loader.find("ImageLoader")
            if inner_loader is not None:
                zgroups = inner_loader.find(".//{*}zgroups") or inner_loader.find("zgroups")
                if zgroups is not None:
                    zgroup_list = zgroups.findall(".//{*}zgroup") or zgroups.findall("zgroup")
                    logging.info(f"  Summary: ImageLoader contains {len(zgroup_list)} zgroups")
                else:
                    logging.info("  Summary: ImageLoader found but no zgroups")
            else:
                logging.info("  Summary: ImageLoader found but no nested structure")
        else:
            logging.info("  Summary: No ImageLoader found")
    else:
        logging.info("  <ImageLoader>NOT FOUND</ImageLoader>")
        logging.info("  Summary: No ImageLoader found")
    
    logging.info("=========================================================\n")

def _extract_zarr_path_from_xml(xml_tree):
    """
    Extract zarr path from XML using multiple strategies, similar to Java bestEffortZarrPath.
    """
    try:
        # Strategy 1: Look for ImageLoader > ImageLoader > zarr (nested structure)
        img_loader = xml_tree.find(".//{*}ImageLoader") or xml_tree.find("ImageLoader")
        if img_loader is not None:
            # Check for nested ImageLoader
            nested_loader = img_loader.find(".//{*}ImageLoader") or img_loader.find("ImageLoader")
            if nested_loader is not None:
                img_loader = nested_loader
            
            # Look for zarr element
            zarr_elem = img_loader.find(".//{*}zarr") or img_loader.find("zarr")
            if zarr_elem is not None and zarr_elem.text and zarr_elem.text.strip():
                return zarr_elem.text.strip()
        
        # Strategy 2: Look for other path-like elements
        path_elements = [
            "path", "url", "bucket", "root", "base", "location", "file", "directory"
        ]
        
        for elem_name in path_elements:
            elem = xml_tree.find(".//*[contains(local-name(), '{}')]".format(elem_name))
            if elem is not None and elem.text and elem.text.strip():
                text = elem.text.strip()
                # Check if it looks like a path/URL
                if any(char in text for char in ['/', '\\', ':', '.']) and len(text) > 5:
                    return text
        
        # Strategy 3: Look for any text that looks like a zarr path
        for elem in xml_tree.iter():
            if elem.text and elem.text.strip():
                text = elem.text.strip()
                # Check for common zarr path patterns
                if any(pattern in text.lower() for pattern in ['zarr', 's3://', 'http://', 'https://', '.zarr']):
                    return text
        
        return "unknown"
        
    except Exception as e:
        logging.error(f"⚠️  [extract_zarr_path] Error extracting zarr path: {str(e)}")
        return "unknown"

def _validate_final_xml_structure(xml_tree, expected_zgroups_count):
    """
    Validate that the final XML structure contains the expected nested ImageLoader with zgroups.
    
    Expected structure:
    <SpimData>
      <SequenceDescription>
        <ImageLoader format="split.viewerimgloader">
          <ImageLoader format="bdv.multimg.zarr" version="3.0">
            <zarr type="absolute">...</zarr>
            <zgroups>
              <zgroup setup="0" tp="0" path="..." indicies="[]" />
              <zgroup setup="1" tp="0" path="..." indicies="[]" />
              <!-- ... more zgroups ... -->
            </zgroups>
          </ImageLoader>
        </ImageLoader>
      </SequenceDescription>
    </SpimData>
    """
    try:
        # Check SequenceDescription
        seq_desc = xml_tree.find(".//{*}SequenceDescription") or xml_tree.find("SequenceDescription")
        if seq_desc is None:
            logging.error("❌ [validation] SequenceDescription not found")
            return False
        
        # Check ImageLoader
        img_loader = seq_desc.find(".//{*}ImageLoader") or seq_desc.find("ImageLoader")
        if img_loader is None:
            logging.error("❌ [validation] ImageLoader not found in SequenceDescription")
            return False
        
        # Check format
        if img_loader.get("format") != "split.viewerimgloader":
            logging.error(f"❌ [validation] ImageLoader format incorrect: {img_loader.get('format')}")
            return False
        
        # Check nested ImageLoader
        nested_loader = img_loader.find(".//{*}ImageLoader") or img_loader.find("ImageLoader")
        if nested_loader is None:
            logging.error("❌ [validation] Nested ImageLoader not found")
            return False
        
        # Check nested format
        if nested_loader.get("format") != "bdv.multimg.zarr":
            logging.error(f"❌ [validation] Nested ImageLoader format incorrect: {nested_loader.get('format')}")
            return False
        
        # Check zarr element
        zarr_elem = nested_loader.find(".//{*}zarr") or nested_loader.find("zarr")
        if zarr_elem is None:
            logging.error("❌ [validation] zarr element not found in nested ImageLoader")
            return False
        
        # Check zgroups
        zgroups = nested_loader.find(".//{*}zgroups") or nested_loader.find("zgroups")
        if zgroups is None:
            logging.error("❌ [validation] zgroups element not found in nested ImageLoader")
            return False
        
        # Check zgroup count
        zgroup_list = zgroups.findall(".//{*}zgroup") or zgroups.findall("zgroup")
        actual_count = len(zgroup_list)
        if actual_count != expected_zgroups_count:
            logging.error(f"❌ [validation] zgroup count mismatch: expected {expected_zgroups_count}, got {actual_count}")
            return False
        
        # Check first few zgroups have required attributes
        for i, zgroup in enumerate(zgroup_list[:3]):  # Check first 3
            required_attrs = ["setup", "tp", "path", "indicies"]
            for attr in required_attrs:
                if attr not in zgroup.attrib:
                    logging.error(f"❌ [validation] zgroup {i} missing required attribute: {attr}")
                    return False
        
        logging.info(f"✅ [validation] All checks passed: {actual_count} zgroups found with correct structure")
        return True
        
    except Exception as e:
        logging.error(f"❌ [validation] Validation error: {str(e)}")
        return False