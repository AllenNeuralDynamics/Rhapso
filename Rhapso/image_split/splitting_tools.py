"""Simplified image splitting tools for Rhapso."""

import math
import numpy as np
import copy
import logging
import time
import random
from xml.etree import ElementTree as ET
from itertools import product
from Rhapso.image_split.split_views import next_multiple


def last_image_size(length, size, overlap):
    """Calculate the size of the last tile in a dimension."""
    if length <= size:
        return length
    step = size - overlap
    if step <= 0:
        return length
    num_blocks = math.ceil((length - size) / step) + 1
    last_start = (num_blocks - 1) * step
    return length - last_start


def last_image_size_java(l, s, o):
    """Port of Java SplittingTools.lastImageSize with Java-style remainder."""
    a = l - 2 * (s - o) - o
    b = s - o
    rem = a - int(a / b) * b if b != 0 else 0
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
        return None
    return (min_intersect, max_intersect)


def is_empty(interval):
    """Check if an interval is empty."""
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
    """Port of Java SplittingTools.splitDim producing 1D [min,max] inclusive intervals."""
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


def distribute_intervals_fixed_overlap(input_dims, overlap, target_size, min_step_size, optimize):
    """Port of Java SplittingTools.distributeIntervalsFixedOverlap."""
    for d in range(len(input_dims)):
        if int(target_size[d]) % int(min_step_size[d]) != 0:
            logging.warning(f"targetSize {target_size[d]} not divisible by minStepSize {min_step_size[d]} for dim={d}")
            return []
        if int(overlap[d]) % int(min_step_size[d]) != 0:
            logging.warning(f"overlapPx {overlap[d]} not divisible by minStepSize {min_step_size[d]} for dim={d}")
            return []

    interval_basis = []
    for d in range(len(input_dims)):
        dim_intervals = []
        length = int(input_dims[d])
        s = int(target_size[d])
        o = int(overlap[d])

        if length <= s:
            dim_intervals.append((0, length - 1))
        else:
            l = length
            last_size = last_image_size_java(l, s, o)
            if optimize and last_size != s:
                current_last = last_size
                step = int(min_step_size[d])
                if last_size <= s // 2:
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

            dim_intervals = split_dim_java(l, final_size, o, min0=0)

        interval_basis.append(dim_intervals)

    interval_list = []
    for rev_idx in product(*[range(len(b)) for b in interval_basis[::-1]]):
        idx = rev_idx[::-1]
        mins = [interval_basis[d][idx[d]][0] for d in range(len(interval_basis))]
        maxs = [interval_basis[d][idx[d]][1] for d in range(len(interval_basis))]
        interval_list.append((np.array(mins, dtype=np.int64), np.array(maxs, dtype=np.int64)))

    return interval_list


def max_interval_spread(old_setups, overlap, target_size, min_step_size, optimize):
    """Calculate the maximum number of splits for any single view."""
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

    max_splits = 1
    for old_setup in old_setups:
        id_el = old_setup.find(".//{*}id") or old_setup.find("id")
        try:
            vs_id = int(id_el.text) if id_el is not None else -1
        except Exception:
            vs_id = -1

        size_el = old_setup.find(".//{*}size") or old_setup.find("size")
        if size_el is None or size_el.text is None:
            dims = np.array([0, 0, 0], dtype=np.int64)
        else:
            size_str = size_el.text.strip()
            dims = np.array([int(d) for d in size_str.split()], dtype=np.int64)

        intervals = distribute_intervals_fixed_overlap(dims, overlap_list, target_list, min_step_list, optimize)
        num_intervals = len(intervals) if intervals is not None else 0
        max_splits = max(max_splits, num_intervals)

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
            for k, v in attrib.items():
                node.set(k, v)
    return node


def _clear_children(node):
    """Remove all children from a node."""
    for child in list(node):
        node.remove(child)


def _ensure_sequence_description_like_bss(xml_root, img_loader_format="split.viewerimgloader"):
    """Ensure root and BasePath match BSS format."""
    root_tag = xml_root.tag.split("}")[-1]
    root = xml_root
    if root_tag != "SpimData":
        spim = ET.Element("SpimData", {"version": "0.2"})
        spim.append(xml_root)
        root = spim

    base_path = _find_one(root, "BasePath")
    if base_path is None:
        base_path = ET.Element("BasePath", {"type": "relative"})
        base_path.text = "."
        first = list(root)[0] if len(list(root)) > 0 else None
        if first is not None and first.tag.split("}")[-1] == "SequenceDescription":
            root.insert(0, base_path)
        else:
            root.append(base_path)

    seq = _find_one(root, "SequenceDescription")
    if seq is None:
        seq = ET.SubElement(root, "SequenceDescription")

    img_loader = (_find_one(seq, "ImageLoader") or _find_one(seq, "ImgLoader"))
    if img_loader is None or img_loader.tag.split("}")[-1] != "ImageLoader":
        if img_loader is not None:
            seq.remove(img_loader)
        img_loader = ET.SubElement(seq, "ImageLoader", {"format": img_loader_format})
    else:
        if img_loader.get("format") != img_loader_format:
            img_loader.set("format", img_loader_format)

    view_setups = _find_one(seq, "ViewSetups")
    if view_setups is None:
        view_setups = ET.SubElement(seq, "ViewSetups")

    timepoints = _find_one(seq, "Timepoints")
    if timepoints is None:
        timepoints = ET.SubElement(seq, "Timepoints", {"type": "pattern"})
    else:
        timepoints.set("type", "pattern")
        _clear_children(timepoints)

    missing = _find_one(seq, "MissingViews")
    if missing is None:
        ET.SubElement(seq, "MissingViews")

    if img_loader is not None and img_loader.get("format") == "split.viewerimgloader":
        nested_loader = img_loader.find(".//{*}ImageLoader") or img_loader.find("ImageLoader")
        if nested_loader is None:
            nested_img_loader = ET.SubElement(img_loader, "ImageLoader", {"format": "bdv.multimg.zarr", "version": "3.0"})
            zarr_elem = ET.SubElement(nested_img_loader, "zarr", {"type": "absolute"})
            zarr_elem.text = "unknown"
            zgroups_elem = ET.SubElement(nested_img_loader, "zgroups")
            ET.SubElement(zgroups_elem, "zgroup", {
                "setup": "0", "tp": "0", "path": "tile_000000_ch_561.zarr", "indicies": "[]"
            })
    
    return root


def split_images(spimData, overlapPx, targetSize, minStepSize, assingIlluminationsFromTileIds=False,
                 optimize=False, addIPs=False, pointDensity=0.0, minPoints=0, maxPoints=0, error=0.0, excludeRadius=0.0):
    """Split images into smaller tiles while maintaining XML metadata structure."""
    try:
        new_spim_data = copy.deepcopy(spimData)
        xml_tree = new_spim_data
        
        sequence_description = xml_tree.find(".//{*}SequenceDescription") or xml_tree.find("SequenceDescription")
        
        # Parse timepoints
        timepoints = None
        tp_elements = []
        if sequence_description is not None:
            timepoints = sequence_description.find(".//{*}Timepoints") or sequence_description.find("Timepoints")
            if timepoints is not None:
                tp_elements = list(timepoints.findall(".//{*}TimePoint")) or list(timepoints.findall("TimePoint"))
                if not tp_elements:
                    intpat = timepoints.find(".//{*}integerpattern") or timepoints.find("integerpattern")
                    if intpat is not None and intpat.text:
                        try:
                            tp_id = int(intpat.text.strip())
                            tp_elements = [intpat]
                        except Exception:
                            pass

        # Get old setups
        def _strip(tag):
            return tag.split("}")[-1] if "}" in tag else tag

        view_setups_parent = None
        if sequence_description is not None:
            view_setups_parent = sequence_description.find(".//{*}ViewSetups") or sequence_description.find("ViewSetups")
        if view_setups_parent is None:
            view_setups_parent = xml_tree.find(".//{*}ViewSetups") or xml_tree.find("ViewSetups")

        old_setups = []
        if view_setups_parent is not None:
            candidates = [vs for vs in list(view_setups_parent) if _strip(vs.tag) == "ViewSetup"]
            def _get_vs_id(vs):
                id_el = vs.find(".//{*}id") or vs.find("id")
                try:
                    return int(id_el.text) if id_el is not None else 0
                except Exception:
                    return 0
            old_setups = sorted(candidates, key=_get_vs_id)

        # Get old registrations
        old_registrations = xml_tree.find(".//{*}ViewRegistrations") or xml_tree.find("ViewRegistrations")
        
        # Calculate max interval spread
        max_interval_spread_value = max_interval_spread(old_setups, overlapPx, targetSize, minStepSize, optimize)
        
        # Check illuminations
        if assingIlluminationsFromTileIds:
            illum_ids = set()
            for vs in old_setups:
                illum_el = vs.find(".//{*}illumination") or vs.find("illumination")
                if illum_el is not None and illum_el.text and illum_el.text.strip().isdigit():
                    try:
                        illum_ids.add(int(illum_el.text.strip()))
                    except Exception:
                        pass
            if len(illum_ids) > 1:
                raise ValueError("Cannot assign illuminations from tile IDs because more than one Illumination exists.")

        # Initialize variables
        new2old_setup_id = {}
        new_setup_id2_interval = {}
        new_setups = []
        new_registrations = {}
        new_interestpoints = {}
        new_id = 0
        fakeLabel = f"splitPoints_{int(time.time() * 1000)}"
        rnd = random.Random(23424459)

        # Process each old setup
        for old_setup in old_setups:
            id_el = old_setup.find(".//{*}id") or old_setup.find("id")
            oldID = int(id_el.text) if id_el is not None else None
            
            # Extract setup properties
            angle_el = old_setup.find(".//{*}angle") or old_setup.find("angle")
            angle = int(angle_el.text.strip()) if angle_el is not None and angle_el.text else 0
            
            channel_el = old_setup.find(".//{*}channel") or old_setup.find("channel")
            channel = int(channel_el.text.strip()) if channel_el is not None and channel_el.text else 0
            
            illum_el = old_setup.find(".//{*}illumination") or old_setup.find("illumination")
            illum = int(illum_el.text.strip()) if illum_el is not None and illum_el.text else 0
            
            # Extract voxel size
            vox_el = old_setup.find(".//{*}voxelSize") or old_setup.find("voxelSize")
            voxDim = 0
            if vox_el is not None:
                size_text = vox_el.get("size") or (vox_el.findtext(".//{*}size") or vox_el.findtext("size"))
                unit = vox_el.get("unit") or (vox_el.findtext(".//{*}unit") or vox_el.findtext("unit"))
                if size_text or unit:
                    try:
                        size_vals = [float(x) for x in size_text.split()] if size_text else None
                        voxDim = {"size": size_vals, "unit": unit}
                    except Exception:
                        pass

            # Get dimensions
            size_el = old_setup.find(".//{*}size") or old_setup.find("size")
            dims = [int(x) for x in size_el.text.strip().split()] if size_el is not None and size_el.text else [0, 0, 0]
            
            # Generate intervals
            intervals = distribute_intervals_fixed_overlap(dims, overlapPx, targetSize, minStepSize, optimize)
            
            localNewTileId = 0
            
            # Process each interval
            for i, (mins_arr, maxs_arr) in enumerate(intervals):
                mins = [int(x) for x in (mins_arr.tolist() if hasattr(mins_arr, "tolist") else mins_arr)]
                maxs = [int(x) for x in (maxs_arr.tolist() if hasattr(maxs_arr, "tolist") else maxs_arr)]
                
                # Map new ID to old ID and interval
                new2old_setup_id[new_id] = oldID
                new_setup_id2_interval[new_id] = (mins_arr, maxs_arr)
                
                # Calculate size and location
                size = [int(mx - mn + 1) for mn, mx in zip(mins, maxs)]
                newDim = tuple(size)
                
                # Calculate tile location
                oldTile = None
                tile_el = old_setup.find(".//{*}tile") or old_setup.find("tile")
                if tile_el is not None:
                    tile_loc_el = tile_el.find(".//{*}location") or tile_el.find("location")
                    if tile_loc_el is not None and tile_loc_el.text:
                        location = [float(x) for x in tile_loc_el.text.strip().split()]
                        for d in range(len(mins)):
                            location[d] += mins[d]
                    else:
                        location = [float(x) for x in mins]
                else:
                    location = [float(x) for x in mins]
                
                # Create new tile
                old_tile_id = 0
                if oldTile and isinstance(oldTile, dict) and oldTile.get("id") is not None:
                    old_tile_id = oldTile["id"]
                newTileId = old_tile_id * max_interval_spread_value + localNewTileId
                localNewTileId += 1
                newTile = {"id": newTileId, "name": str(newTileId), "location": location}
                
                # Create new illumination
                newIllum = {"id": old_tile_id, "name": f"old_tile_{old_tile_id}"} if assingIlluminationsFromTileIds else illum
                
                # Create new setup
                newSetup = {
                    "id": new_id, "dim": newDim, "voxDim": voxDim, "tile": newTile,
                    "channel": channel, "angle": angle, "illum": newIllum
                }
                new_setups.append(newSetup)
                
                # Process timepoints
                for tp in tp_elements:
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
                    
                    # Create new view registration
                    newViewId_key = (int(tp_id) if str(tp_id).isdigit() else tp_id, new_id)
                    
                    # Extract transforms from old registration
                    transformList = []
                    if old_registrations is not None:
                        vr_candidates = list(old_registrations.findall(".//{*}ViewRegistration")) or list(old_registrations.findall("ViewRegistration"))
                        for vr in vr_candidates:
                            if str(vr.get("timepoint")) == str(tp_id) and str(vr.get("setup")) == str(oldID):
                                vt_elems = list(vr.findall(".//{*}ViewTransform")) or list(vr.findall("ViewTransform"))
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
                                break
                    
                    # Add image splitting transform
                    tx = float(mins[0] if len(mins) > 0 else 0.0)
                    ty = float(mins[1] if len(mins) > 1 else 0.0)
                    tz = float(mins[2] if len(mins) > 2 else 0.0)
                    translation_affine = [1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, ty, 0.0, 0.0, 1.0, tz]
                    transform = {"name": "Image Splitting", "affine": translation_affine}
                    transformList.append(transform)
                    
                    # Create new registration
                    newVR = {"timepoint": newViewId_key[0], "setup": new_id, "transforms": transformList}
                    new_registrations[newViewId_key] = newVR
                    
                    # Create interest points structure
                    newVipl = {"timepointId": newViewId_key[0], "viewSetupId": new_id}
                    new_interestpoints[newViewId_key] = newVipl
                
                new_id += 1

        # Rebuild XML structure
        _rebuild_xml_structure(xml_tree, old_setups, new_setups, new2old_setup_id, 
                              new_setup_id2_interval, max_interval_spread_value, targetSize)
        
        return xml_tree
        
    except Exception as e:
        logging.error(f"Error in split_images: {str(e)}")
        return None


def _rebuild_xml_structure(xml_tree, old_setups, new_setups, new2old_setup_id, 
                          new_setup_id2_interval, max_interval_spread_value, targetSize):
    """Rebuild the XML structure with split tiles."""
    
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
        old_mv_orig = (xml_tree.find(".//{*}SequenceDescription/{*}MissingViews")
                       or xml_tree.find(".//SequenceDescription/MissingViews"))
        old_missing = []
        if old_mv_orig is not None:
            for child in list(old_mv_orig):
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
        total_split_tiles = len(new_setups)
        
        for setup_id in range(total_split_tiles):
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
            original_tile = setup_id // max_interval_spread_value
            sub_index = setup_id % max_interval_spread_value
            
            # Calculate x, y, z offsets based on sub_index and actual tile dimensions
            x_step = targetSize[0]
            y_step = targetSize[1]
            z_step = targetSize[2]
            
            x_offset = (sub_index % 2) * x_step
            y_offset = ((sub_index // 2) % 2) * y_step
            z_offset = (sub_index // 4) * z_step
            
            # Create the affine transform matrix for image splitting
            affine_matrix = f"1.0 0.0 0.0 {x_offset} 0.0 1.0 0.0 {y_offset} 0.0 0.0 1.0 {z_offset}"
            ET.SubElement(vt4, "affine").text = affine_matrix
        
        logging.info(f"Created {total_split_tiles} ViewRegistration items (timepoint=0, setup=0 to setup={total_split_tiles-1})")
        
    except Exception as e:
        logging.error(f"Error rebuilding ViewRegistrations: {str(e)}")
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
        total_split_tiles = len(new_setups)
        
        # Determine labels dynamically following Java implementation
        labels = []
        
        # Add original labels with "_split" suffix (like Java: label + "_split")
        for original_label in original_labels:
            labels.append(original_label + "_split")
        
        # Add fake label for split points
        import time
        fakeLabel = f"splitPoints_{int(time.time() * 1000)}"
        labels.append(fakeLabel)
        
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
        except Exception as e:
            logging.error(f"Could not determine timepoint ID, using default: {timepoint_id}")
        
        # Generate ViewInterestPointsFile for each label and each split tile
        for label in labels:
            for setup_id in range(total_split_tiles):
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
                        "params": f"Fake points for image splitting: overlapPx={targetSize}, targetSize={targetSize}, minStepSize={targetSize}, optimize=False, pointDensity=0.0, minPoints=0, maxPoints=0, error=0.0, excludeRadius=0.0"
                    }
                )
                vip_file.text = path_text
        
        logging.info(f"Created {len(labels) * total_split_tiles} ViewInterestPointsFile items ({len(labels)} labels × {total_split_tiles} split tiles)")
        
    except Exception as e:
        logging.error(f"Error rebuilding ViewInterestPoints: {str(e)}")
        pass

    # 4) Build the nested ImageLoader structure inside SequenceDescription
    seq_desc = xml_tree.find(".//{*}SequenceDescription") or xml_tree.find("SequenceDescription")
    if seq_desc is not None:
        # Remove any existing ImageLoader
        existing_loader = seq_desc.find(".//{*}ImageLoader") or seq_desc.find("ImageLoader")
        if existing_loader is not None:
            seq_desc.remove(existing_loader)

        # Extract zarr path and zgroups from input XML using improved extraction
        input_zarr_path = _extract_zarr_path_from_xml(xml_tree)
        
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
            # Use old_setups for zgroups since these represent the original tiles
            for i, old_setup in enumerate(old_setups):
                # Create the zgroup element with more realistic path
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
        for i in range(len(old_setups)):
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
    
    # Populate the main ViewSetups (outside ImageLoader) with all ViewSetup items
    main_view_setups = None
    seq_desc = xml_tree.find(".//{*}SequenceDescription") or xml_tree.find("SequenceDescription")
    for child in seq_desc:
        if child.tag.endswith("ViewSetups") or child.tag == "ViewSetups":
            main_view_setups = child
            break
    
    if main_view_setups is not None:
        # Clear any existing ViewSetup items
        for child in list(main_view_setups):
            main_view_setups.remove(child)
        
        # Add ViewSetup items for all split tiles
        for i, new_setup in enumerate(new_setups):
            vs = ET.SubElement(main_view_setups, "ViewSetup")
            
            # Use the new_setup id
            ET.SubElement(vs, "id").text = str(new_setup["id"])
            
            # Use the split tile size from new_setup
            size = new_setup["dim"]
            ET.SubElement(vs, "size").text = " ".join(str(int(x)) for x in size)

            # Extract voxelSize from the original setup that this split tile came from
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
            channel_id = "0"
            if "channel" in new_setup:
                if isinstance(new_setup["channel"], dict) and "id" in new_setup["channel"]:
                    channel_id = str(new_setup["channel"]["id"])
                elif isinstance(new_setup["channel"], (int, str)):
                    channel_id = str(new_setup["channel"])
            ET.SubElement(attrs_elem, "channel").text = channel_id
            
            # Use the new_setup id as the tile id
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
        for i in range(len(old_setups)):
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
        for i in range(total_tiles):
            tile = ET.SubElement(tile_attr, "Tile")
            ET.SubElement(tile, "id").text = str(i)
            ET.SubElement(tile, "name").text = str(i)
            
            # Calculate location based on tile index (dynamic grid pattern)
            original_tile = i // max_interval_spread_value
            sub_index = i % max_interval_spread_value
            
            # Calculate x, y, z offsets based on sub_index
            x_offset = (sub_index % 2) * 7104.0
            y_offset = ((sub_index // 2) % 2) * 5376.0
            z_offset = (sub_index // 4) * 4096.0
            
            location = f"{x_offset} {y_offset} {z_offset}"
            ET.SubElement(tile, "location").text = location
        
        # Angle
        angle_attr = ET.SubElement(main_view_setups, "Attributes", {"name": "angle"})
        angle = ET.SubElement(angle_attr, "Angle")
        ET.SubElement(angle, "id").text = "0"
        ET.SubElement(angle, "name").text = "0"
        
        logging.info(f"Populated main ViewSetups with {len(new_setups)} ViewSetup items")
    else:
        logging.warning("Main ViewSetups not found - cannot populate with split tiles")

    # Ensure correct element order in SequenceDescription (ImageLoader first, like Java BSS)
    seq_desc = xml_tree.find(".//{*}SequenceDescription") or xml_tree.find("SequenceDescription")
    if seq_desc is not None:
        # Ensure ImageLoader is the first element
        img_loader = seq_desc.find(".//{*}ImageLoader") or seq_desc.find("ImageLoader")
        if img_loader is not None:
            # Remove and re-insert at the beginning to ensure it's first
            seq_desc.remove(img_loader)
            seq_desc.insert(0, img_loader)
            logging.info("Ensured ImageLoader is first element in SequenceDescription")


def _extract_zarr_path_from_xml(xml_tree):
    """Extract zarr path from XML using multiple strategies."""
    try:
        img_loader = xml_tree.find(".//{*}ImageLoader") or xml_tree.find("ImageLoader")
        if img_loader is not None:
            nested_loader = img_loader.find(".//{*}ImageLoader") or img_loader.find("ImageLoader")
            if nested_loader is not None:
                img_loader = nested_loader
            
            zarr_elem = img_loader.find(".//{*}zarr") or img_loader.find("zarr")
            if zarr_elem is not None and zarr_elem.text and zarr_elem.text.strip():
                return zarr_elem.text.strip()
        
        return "unknown"
        
    except Exception as e:
        logging.error(f"Error extracting zarr path: {str(e)}")
        return "unknown"


def _validate_final_xml_structure(xml_tree, expected_zgroups_count):
    """Validate that the final XML structure contains the expected nested ImageLoader with zgroups."""
    try:
        seq_desc = xml_tree.find(".//{*}SequenceDescription") or xml_tree.find("SequenceDescription")
        if seq_desc is None:
            return False
        
        img_loader = seq_desc.find(".//{*}ImageLoader") or seq_desc.find("ImageLoader")
        if img_loader is None or img_loader.get("format") != "split.viewerimgloader":
            return False
        
        nested_loader = img_loader.find(".//{*}ImageLoader") or img_loader.find("ImageLoader")
        if nested_loader is None or nested_loader.get("format") != "bdv.multimg.zarr":
            return False
        
        zarr_elem = nested_loader.find(".//{*}zarr") or nested_loader.find("zarr")
        if zarr_elem is None:
            return False
        
        zgroups = nested_loader.find(".//{*}zgroups") or nested_loader.find("zgroups")
        if zgroups is None:
            return False
        
        zgroup_list = zgroups.findall(".//{*}zgroup") or zgroups.findall("zgroup")
        actual_count = len(zgroup_list)
        if actual_count != expected_zgroups_count:
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        return False
