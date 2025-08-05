# splitting_tools.py

import math
import numpy as np
import copy
from xml.etree import ElementTree as ET
from itertools import product
from Rhapso.image_split.SplitViews import closest_larger_long_divisable_by

def last_image_size(length, size, overlap):
    """Calculates the size of the last tile in a dimension."""
    if length <= size:
        return length
    
    step = size - overlap
    if step <= 0:
         return length # Avoid infinite loops
         
    num_blocks = math.ceil((length - size) / step) + 1
    
    last_start = (num_blocks - 1) * step
    return length - last_start


def distribute_intervals_fixed_overlap(input_dims, overlap, target_size, min_step_size, optimize):
    """
    Computes a set of overlapping intervals for each dimension.
    """
    interval_basis = []
    input_mins = np.zeros(len(input_dims), dtype=np.int64)

    for d in range(len(input_dims)):
        dim_intervals = []
        length = input_dims[d]
        s = target_size[d]
        o = overlap[d]
        
        if length <= s:
            dim_intervals.append((input_mins[d], input_mins[d] + length - 1))
        else:
            final_size = s
            # Simplified optimization logic
            if optimize:
                best_size = s
                min_remainder = float('inf')
                # Search for a better tile size to minimize the last tile's irregularity
                for test_s in range(int(s * 0.8), int(s * 1.2)):
                     test_s = closest_larger_long_divisable_by(test_s, min_step_size[d])
                     if test_s <= o: continue
                     rem = (length - test_s) % (test_s - o)
                     if rem < min_remainder:
                         min_remainder = rem
                         best_size = test_s
                final_size = best_size

            # Split dimension based on final_size and overlap
            start = input_mins[d]
            while start < input_mins[d] + length:
                end = min(start + final_size - 1, input_mins[d] + length - 1)
                dim_intervals.append((start, end))
                if end >= input_mins[d] + length - 1:
                    break
                start = end - o + 1

        interval_basis.append(dim_intervals)
    
    # Create combined intervals from per-dimension lists
    interval_list = []
    for combined in product(*interval_basis):
        mins = [iv[0] for iv in combined]
        maxs = [iv[1] for iv in combined]
        interval_list.append((np.array(mins), np.array(maxs)))
        
    return interval_list

def max_interval_spread(old_setups, overlap, target_size, min_step_size, optimize):
    """Calculates the maximum number of splits for any single view."""
    max_splits = 1
    for old_setup in old_setups:
        size_str = old_setup.find('size').text
        dims = np.array([int(d) for d in size_str.split()])
        intervals = distribute_intervals_fixed_overlap(dims, overlap, target_size, min_step_size, optimize)
        max_splits = max(max_splits, len(intervals))
    return max_splits

def split_images(root, overlap, target_size, min_step_size, assign_illuminations, optimize, **kwargs):
    """
    Splits the images in the XML, returning a new XML root.
    """
    new_root = copy.deepcopy(root)
    
    # Clear existing view-related data
    new_root.find('SequenceDescription/ViewSetups').clear()
    new_root.find('ViewRegistrations').clear()
    if new_root.find('ViewInterestPoints') is not None:
        new_root.find('ViewInterestPoints').clear()
    if new_root.find('SequenceDescription/MissingViews') is not None:
        new_root.find('SequenceDescription/MissingViews').clear()

    old_setups = sorted(
        root.findall('SequenceDescription/ViewSetups/ViewSetup'),
        key=lambda vs: int(vs.find('id').text)
    )
    
    timepoints = root.find('SequenceDescription/Timepoints').findall('TimePoint')
    max_spread = max_interval_spread(old_setups, overlap, target_size, min_step_size, optimize)
    
    new_id = 0
    new_to_old_map = {}

    for old_setup in old_setups:
        old_id = int(old_setup.find('id').text)
        old_tile_id = int(old_setup.find('Tile/id').text)
        size = np.array([int(d) for d in old_setup.find('size').text.split()])
        
        intervals = distribute_intervals_fixed_overlap(size, overlap, target_size, min_step_size, optimize)
        
        local_new_tile_id = 0
        for interval_min, interval_max in intervals:
            new_dims = interval_max - interval_min + 1
            
            # Create new ViewSetup
            vs_parent = new_root.find('SequenceDescription/ViewSetups')
            new_vs = ET.SubElement(vs_parent, 'ViewSetup')
            ET.SubElement(new_vs, 'id').text = str(new_id)
            ET.SubElement(new_vs, 'name').text = f"ViewSetup id={new_id} (from {old_id})"
            ET.SubElement(new_vs, 'size').text = " ".join(map(str, new_dims))
            new_vs.append(copy.deepcopy(old_setup.find('voxelSize')))
            
            # Create new Tile
            new_tile = ET.SubElement(new_vs, 'Tile')
            new_tile_id_val = old_tile_id * max_spread + local_new_tile_id
            ET.SubElement(new_tile, 'id').text = str(new_tile_id_val)
            
            # Create new illumination if requested
            if assign_illuminations:
                illum = ET.SubElement(new_vs, 'Illumination')
                ET.SubElement(illum, 'id').text = str(old_tile_id)
                ET.SubElement(illum, 'name').text = f"old_tile_{old_tile_id}"
            else:
                new_vs.append(copy.deepcopy(old_setup.find('Illumination')))

            new_vs.append(copy.deepcopy(old_setup.find('Channel')))
            new_vs.append(copy.deepcopy(old_setup.find('Angle')))
            
            # Create ViewRegistrations for each timepoint
            vr_parent = new_root.find('ViewRegistrations')
            for tp in timepoints:
                tp_id = tp.get('id')
                new_vr = ET.SubElement(vr_parent, 'ViewRegistration', {'timepoint': tp_id, 'setup': str(new_id)})
                
                # Copy original transforms
                old_vr = root.find(f"ViewRegistrations/ViewRegistration[@timepoint='{tp_id}'][@setup='{old_id}']")
                if old_vr:
                    for vt in old_vr.findall('ViewTransform'):
                        new_vr.append(copy.deepcopy(vt))

                # Add splitting translation transform
                translation = np.eye(4)
                translation[:3, 3] = interval_min
                transform_matrix_str = " ".join(map(str, translation.flatten()[:12]))
                
                new_vt = ET.SubElement(new_vr, 'ViewTransform', {'type': 'affine'})
                ET.SubElement(new_vt, 'name').text = 'Image Splitting'
                ET.SubElement(new_vt, 'affine').text = transform_matrix_str
            
            new_to_old_map[new_id] = old_id
            new_id += 1
            local_new_tile_id += 1
            
    # Note: Interest point splitting is complex and omitted for this conversion.
    # A full implementation would require KD-trees and careful coordinate transformation.
    # Fake interest point generation is also omitted.

    return new_root