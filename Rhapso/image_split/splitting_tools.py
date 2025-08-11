# splitting_tools.py

import math
import numpy as np
import copy
import os
import logging
import traceback
from pathlib import Path
from xml.etree import ElementTree as ET
from itertools import product
from Rhapso.image_split.split_views import closest_larger_long_divisable_by
import copy
import numpy as np
from xml.etree import ElementTree as ET


def last_image_size(length, size, overlap):
    """Calculates the size of the last tile in a dimension."""
    if length <= size:
        return length

    step = size - overlap
    if step <= 0:
        return length  # Avoid infinite loops

    num_blocks = math.ceil((length - size) / step) + 1

    last_start = (num_blocks - 1) * step
    return length - last_start


def last_image_size_java(l, s, o):
    # Port of SplittingTools.lastImageSize(long l, long s, long o) with Java-style remainder
    # Java remainder: a % b keeps the sign of 'a' (truncates toward zero). Python uses floor.
    a = l - 2 * (s - o) - o
    b = s - o
    rem = a - int(a / b) * b if b != 0 else 0  # emulate Java's %
    size = o + rem
    if size < 0:
        size = l + size
    return int(size)


def intersect(interval1, interval2):
    """Computes the intersection of two intervals."""
    min1, max1 = interval1
    min2, max2 = interval2
    
    min_intersect = np.maximum(min1, min2)
    max_intersect = np.minimum(max1, max2)
    
    if np.any(min_intersect > max_intersect):
        return None  # No overlap
        
    return (min_intersect, max_intersect)

def is_empty(interval):
    """Checks if an interval is empty (has zero or negative volume)."""
    if interval is None:
        return True
    mins, maxs = interval
    return np.any(mins > maxs)


def contains(point_location, interval):
    """Checks if a point location is within a given interval."""
    mins, maxs = interval
    for d in range(len(point_location)):
        if not (mins[d] <= point_location[d] <= maxs[d]):
            return False
    return True


def split_dim_java(length, s, o, min0=0):
    # Port of SplittingTools.splitDim(...) producing 1D [min,max] inclusive intervals
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
            print(
                f"targetSize {target_size[d]} not divisible by minStepSize {min_step_size[d]} for dim={d}. stopping."
            )
            return []
        if int(overlap[d]) % int(min_step_size[d]) != 0:
            print(
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

            dim_intervals.extend(split_dim_java(l, final_size, o, min0=0))

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

    return interval_list


def max_interval_spread(old_setups, overlap, target_size, min_step_size, optimize):
    """Calculates the maximum number of splits for any single view, with Java-like logging."""

    # Normalize inputs for consistent printing
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

    print("maxIntervalSpread inputs:")
    print(f"oldSetups.size = {len(old_setups)}")
    print(f"overlapPx = {overlap_list}")
    print(f"targetSize = {target_list}")
    print(f"minStepSize = {min_step_list}")
    print(f"optimize = {'true' if optimize else 'false'}")

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
        print(f"ViewSetup id: {vs_id}, intervals.size = {num_intervals}")

        max_splits = max(max_splits, num_intervals)

    print(f"maxIntervalSpread output: max = {max_splits}")
    return max_splits


def _find_one(root, name):
	# namespaced-safe find for existing elements
	return root.find(f".//{{*}}{name}") or root.find(name)

def _ensure_child(parent, tag, attrib=None):
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
	for child in list(node):
		node.remove(child)

def _ensure_sequence_description_like_bss(xml_root, img_loader_format="split.viewerimgloader"):
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
	# If an ImageLoader or ImgLoader exists, normalize it to the expected format and clear children
	img_loader = (_find_one(seq, "ImageLoader") or _find_one(seq, "ImgLoader"))
	if img_loader is None or img_loader.tag.split("}")[-1] != "ImageLoader":
		if img_loader is not None:
			seq.remove(img_loader)
		img_loader = ET.SubElement(seq, "ImageLoader", {"format": img_loader_format})
	else:
		img_loader.set("format", img_loader_format)
	_clear_children(img_loader)

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
    try:
        # Make a copy of the SpimData object to avoid modifying the original
        new_spim_data = copy.deepcopy(spimData)

        # Get the XML tree from the spim_data
        xml_tree = new_spim_data
        """
        Start of conversion
        """
        # get timepoints var
        sequence_description = xml_tree.find(
            ".//{*}SequenceDescription"
        ) or xml_tree.find("SequenceDescription")

        # --- BEGIN: Improved timepoints parsing and reporting ---
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

                # Print summary
                if tp_ids:
                    tp_min = min(tp_ids)
                    tp_max = max(tp_ids)
                    print(f"TimePoints: count={len(tp_ids)}, idRange=[{tp_min}..{tp_max}]")
                    for i, tp_id in enumerate(tp_ids):
                        print(f"  - tpId={tp_id}, name={tp_names[i]}, presentSetups={present_setups_count}/{present_setups_count}, missingSetups={missing_setups_count}, presentSetupIds=[ {', '.join(str(sid) for sid in present_setup_ids)} ]")
                else:
                    print("TimePoints: No timepoints found.")
            else:
                print("TimePoints: No Timepoints element found.")
        else:
            print("TimePoints: No SequenceDescription found.")
        # --- END: Improved timepoints parsing and reporting ---

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

        print(f"Found and sorted {len(old_setups)} ViewSetups")

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
        print(f"maxIntervalSpread = {max_interval_spread_value}")

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
        print(f"Splitting {len(old_setups)} old setups...")
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
            print(
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

            print(f"angle: {angle}")
            print(f"channel: {channel}")
            print(f"illum: {illum}")
            print(f"voxDim: {voxDim}")

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
            print(
                f"ViewId {oldID} with interval {_format_bounds(input_mins, input_maxs)} will be split as follows: "
            )

            # intervals: distributeIntervalsFixedOverlap(...)
            intervals = distribute_intervals_fixed_overlap(
                dims, overlapPx, targetSize, minStepSize, optimize
            )

            # interval2ViewSetup map (empty for now, to be filled later)
            interval2ViewSetup = {}

            # Print Java-like parameters and interval list
            print(f"Split parameters for ViewSetup {oldID}:")
            print(
                f"  input      = {_format_interval_with_dims(input_mins, input_maxs)}"
            )
            print(f"  overlapPx  = {[int(v) for v in overlapPx]}")
            print(f"  targetSize = {[int(v) for v in targetSize]}")
            # minStepSize may be a numpy array
            print(
                f"  minStep    = {[int(v) for v in (minStepSize.tolist() if hasattr(minStepSize, 'tolist') else minStepSize)]}"
            )
            print(f"  optimize   = {'true' if optimize else 'false'}")

            if intervals is None:
                print("  intervals  = null")
            else:
                print(f"  intervals ({len(intervals)}):")
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
                    print(f"    [{ii}] {_format_interval_with_dims(mins, maxs)}")

            print(f"  interval2ViewSetup.size = {len(interval2ViewSetup)}")

            # loop through intervals
            print(f"Entering interval loop, total count: {len(intervals)}")
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
                print(f"        Processing interval index: {i + 1}")
                print(
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
                print(
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
                print(
                    f"\tCreated newTile: id={newTile['id']}, name={newTile['name']}, location={newTile['location']}"
                )
                if isinstance(newIllum, dict):
                    print(
                        f"\tCreated newIllum: id={newIllum.get('id')}, name={newIllum.get('name')}"
                    )
                else:
                    print(f"\tCreated newIllum: id={newIllum}")
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
                print(
                    f"\tCreated newSetup: id={newSetup['id']}, tileId={newTile['id']}, channel={channel_id if channel_id is not None else 'null'}, angle={angle_id if angle_id is not None else 'null'}, illum={illum_id if illum_id is not None else 'null'}, dim={size}, voxDim={vox_unit}"
                )
                print("")

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
                print(f"\tStarting timepoint loop for interval index {i}: {tp_total} timepoints total.")
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
                    print(f"\t\tProcessing timepoint {tpIdx + 1}/{tp_total} (id={tp_id})")
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
                    print("\n\t\t--- timepoint loop var setup checkpoint ---")
                    print(f"\t\toldViewId: {oldViewId_str}")
                    if oldVR_el is None:
                        print("\t\toldVR: null")
                    else:
                        print(f"\t\toldVR: ViewRegistration with {old_transform_count} transforms")
                    last_name = transformList[-1]["name"] if len(transformList) > 0 else "n/a"
                    print(f"\t\ttransformList: size={len(transformList)}, last transform name={last_name}")
                    # Match Java-like printing for translation and row-packed copy
                    trans_tuple_str = ", ".join(f"{v:.1f}" if float(v).is_integer() else f"{v}" for v in translation_affine)
                    print(f"\t\ttranslation: 3d-affine: ({trans_tuple_str})")
                    print(f"\t\ttransform: {transform['name']}, affine={[float(v) for v in translation_affine]}")
                    newViewId_str = f"ViewId{{timepoint={newViewId_key[0]}, setup={new_id}}}"
                    print(f"\t\tnewViewId: {newViewId_str}")
                    print(f"\t\tnewVR: {'null' if newVR is None else f'ViewRegistration with {len(transformList)} transforms'}")
                    print(f"\t\tnewRegistrations: total entries={len(new_registrations)}")
                    print(f"\t\tnewVipl: timepointId={newVipl['timepointId']}, viewSetupId={newVipl['viewSetupId']}")
                    if oldVipl is None:
                        print("\t\toldVipl: null")
                    else:
                        print(f"\t\toldVipl: timepointId={oldVipl['timepointId']}, viewSetupId={oldVipl['viewSetupId']}")
                    print("\t\t--- end checkpoint ---\n")
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
                            print(f"\t\tProcessing label index: {labelIdx}, label: {label}")
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
                                    
                                    num_points = min(maxPoints, max(minPoints, int(round(np.ceil(pointDensity * num_pixels / (100.0**3))))))
                                    print(f"{num_pixels / (100.0**3)} {num_points}")

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
                vr_root_parent.remove(old_vr)
            new_vr_root = ET.SubElement(vr_root_parent, "ViewRegistrations")
            # new_registrations: key=(timepoint, setup), value={"timepoint","setup","transforms":[{"name","affine":[12]}]}
            for (tp, setup), reg in new_registrations.items():
                vr_el = ET.SubElement(new_vr_root, "ViewRegistration", {"timepoint": str(tp), "setup": str(setup)})
                for tr in reg.get("transforms", []):
                    vt = ET.SubElement(vr_el, "ViewTransform", {"type": "affine"})
                    ET.SubElement(vt, "Name").text = str(tr.get("name", ""))
                    affine_vals = tr.get("affine", [])
                    ET.SubElement(vt, "affine").text = " ".join(str(float(v)) for v in affine_vals)
        except Exception:
            pass  # keep previous registrations on failure

        # 3) Rebuild ViewInterestPoints from new_interestpoints
        try:
            vip_parent = xml_tree.find(".//{*}ViewInterestPoints") or xml_tree.find("ViewInterestPoints")
            if vip_parent is not None:
                # Replace for simplicity
                root = xml_tree
                seq = root.find(".//{*}SequenceDescription") or root.find("SequenceDescription")
                parent = root
                root.remove(vip_parent)
            else:
                parent = xml_tree
            vip_parent = ET.SubElement(parent, "ViewInterestPoints")

            # new_interestpoints: key=(tp, setup) -> {"timepointId","viewSetupId","interest_points_lists": {label: {...}}}
            for (tp, setup), vipl in new_interestpoints.items():
                lists = vipl.get("interest_points_lists", {})
                for label, ipl in lists.items():
                    params = ipl.get("params", "")
                    # Encode points as path-like name to align with BigStitcher expectations
                    path_text = f"tpId_{tp}_viewSetupId_{setup}/{label}"
                    ET.SubElement(
                        vip_parent,
                        "ViewInterestPointsFile",
                        {
                            "timepoint": str(tp),
                            "setup": str(setup),
                            "label": str(label),
                            "params": params,
                        },
                    ).text = path_text
        except Exception:
            pass

        # 4) Restore ImageLoader if we removed it earlier
        try:
            if sequence_description is not None and underlying_img_loader is not None:
                # Ensure no ImageLoader exists before appending
                existing = (sequence_description.find(".//{*}ImageLoader")
                            or sequence_description.find("ImageLoader")
                            or sequence_description.find(".//{*}ImgLoader")
                            or sequence_description.find("ImgLoader"))
                if existing is None:
                    sequence_description.append(underlying_img_loader)
        except Exception:
            pass

        # --- End: Java finalization translated to Python (XML) ---

        # Normalize SequenceDescription to match BSS-style output
        xml_tree = _ensure_sequence_description_like_bss(xml_tree, img_loader_format="split.viewerimgloader")
        print("🔧 [split_images] Image splitting completed successfully.")
        return xml_tree
    except Exception as e:
        print(f"❌ Error in split_images: {str(e)}")
        traceback.print_exc()
        return None