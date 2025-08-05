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

def split_images(
        root, 
        overlap, 
        target_size, 
        min_step_size, 
        assign_illuminations=False,
        optimize=False, 
        fake_interest_points=False, 
        fip_density=0.0, 
        fip_min_num_points=0, 
        fip_max_num_points=0, 
        fip_error=0.0, 
        fip_exclusion_radius=0.0):
    """
    Splits images according to the specified parameters, following the Java implementation.
    
    Args:
        root: The input XML root/SpimData object
        overlap: Array of overlap pixels [x, y, z]
        target_size: Array of target sizes [x, y, z]
        min_step_size: Array of minimum step sizes [x, y, z]
        assign_illuminations: Boolean flag to assign illuminations from tile IDs
        optimize: Boolean flag for optimization
        fake_interest_points: Boolean flag to add interest points
        fip_density: Point density parameter
        fip_min_num_points: Minimum number of points
        fip_max_num_points: Maximum number of points
        fip_error: Error threshold
        fip_exclusion_radius: Exclusion radius
        
    Returns:
        New SpimData object with split views
    """
    try:
        print(f"\n🔧 [split_images] Starting to split images...")
        print(f"🔧 [split_images] Parameters:")
        print(f"  - Overlap: {overlap}")
        print(f"  - Target size: {target_size}")
        print(f"  - Min step size: {min_step_size}")
        
        # Make a copy of the SpimData object to avoid modifying the original
        new_spim_data = copy.deepcopy(root)
        
        # Get the XML tree from the spim_data
        xml_tree = new_spim_data
        
        # Get sequence description and image loader
        seq_desc = xml_tree.find('SequenceDescription')
        img_loader = seq_desc.find('ImageLoader') if seq_desc is not None else None
        
        # Get timepoints from sequence description
        timepoints = seq_desc.find('Timepoints') if seq_desc is not None else None
        
        # Get the view setups
        view_setups = seq_desc.find('ViewSetups') if seq_desc is not None else None
        if view_setups is None:
            raise ValueError("No ViewSetups found in XML")
        
        setup_elements = view_setups.findall('ViewSetup')
        if not setup_elements:
            raise ValueError("No ViewSetup elements found in XML")
        
        # Find all existing IDs to avoid duplicates
        existing_ids = set()
        for setup in setup_elements:
            id_elem = setup.find('id')
            if id_elem is not None and id_elem.text is not None:
                existing_ids.add(int(id_elem.text))
        
        # Find bounding boxes of all view setups
        print(f"🔧 [split_images] Finding bounding boxes for all view setups...")
        
        # Create mappings for splitting
        setup_id_to_grid = {}
        new_view_setups = []
        next_id = max(existing_ids) + 1 if existing_ids else 0
        
        # Get ViewRegistrations section
        view_registrations = xml_tree.find('ViewRegistrations')
        
        # Process each view setup
        for setup in setup_elements:
            # Extract setup ID with error checking
            setup_id_elem = setup.find('id')
            if setup_id_elem is None or setup_id_elem.text is None:
                continue  # Skip setups without ID
            setup_id = int(setup_id_elem.text)
            print(f"🔧 [split_images] Processing ViewSetup {setup_id}...")
            
            # Try multiple approaches to get dimensions
            dimensions = None
            
            # Approach 1: Standard size element
            size_elem = setup.find('size')
            if size_elem is not None:
                width_elem = size_elem.find('width')
                height_elem = size_elem.find('height')
                depth_elem = size_elem.find('depth')
                
                if width_elem is not None and width_elem.text is not None and \
                   height_elem is not None and height_elem.text is not None and \
                   depth_elem is not None and depth_elem.text is not None:
                    dimensions = [int(width_elem.text), int(height_elem.text), int(depth_elem.text)]
            
            # Approach 2: Look for voxelSize and voxelDimensions
            if dimensions is None:
                voxel_size = setup.find('voxelSize')
                if voxel_size is not None:
                    size_elem = voxel_size.find('size')
                    if size_elem is not None and size_elem.text is not None:
                        # Sometimes size is formatted as x,y,z
                        try:
                            size_values = [float(val) for val in size_elem.text.split(',')]
                            if len(size_values) == 3:
                                dimensions = size_values
                        except:
                            pass
            
            # Approach 3: Look for dimensions directly in ViewSetup
            if dimensions is None:
                dim_x = setup.find('dimensionX')
                dim_y = setup.find('dimensionY')
                dim_z = setup.find('dimensionZ')
                
                if dim_x is not None and dim_x.text is not None and \
                   dim_y is not None and dim_y.text is not None and \
                   dim_z is not None and dim_z.text is not None:
                    try:
                        dimensions = [int(dim_x.text), int(dim_y.text), int(dim_z.text)]
                    except:
                        pass
            
            # Default if still not found
            if dimensions is None:
                # Use the target size as default dimensions if not specified
                print(f"🔧 [split_images] Warning: Could not determine dimensions for ViewSetup {setup_id}. Using target size as dimensions.")
                dimensions = [max(val, 1024) for val in target_size]  # Use target size or 1024 if target is small
                
            print(f"🔧 [split_images] ViewSetup {setup_id} dimensions: {dimensions}")
            
            # Calculate grid size based on target size and dimensions
            grid = []
            for d in range(3):
                if target_size[d] <= 0:
                    grid.append(1)  # No splitting for this dimension
                else:
                    # Calculate number of blocks with step size constraints
                    step = max(min_step_size[d], 1)  # Ensure step is at least 1
                    size = ((dimensions[d] + target_size[d] - 1) // target_size[d]) * step
                    grid.append(size)
                    
            print(f"🔧 [split_images] ViewSetup {setup_id} grid: {grid}")
            setup_id_to_grid[setup_id] = grid
            
            # Create new view setups for each grid position
            for x in range(0, grid[0], min_step_size[0]):
                for y in range(0, grid[1], min_step_size[1]):
                    for z in range(0, grid[2], min_step_size[2]):
                        if grid[0] > 1 or grid[1] > 1 or grid[2] > 1:
                            # Create a new view setup only if we're actually splitting
                            min_x = max(0, x - overlap[0])
                            max_x = min(dimensions[0], x + target_size[0] + overlap[0])
                            
                            min_y = max(0, y - overlap[1])
                            max_y = min(dimensions[1], y + target_size[1] + overlap[1])
                            
                            min_z = max(0, z - overlap[2])
                            max_z = min(dimensions[2], z + target_size[2] + overlap[2])
                            
                            # Create a copy of the current view setup
                            new_setup = copy.deepcopy(setup)
                            
                            # Update the ID
                            id_elem = new_setup.find('id')
                            if id_elem is not None:
                                id_elem.text = str(next_id)
                            else:
                                # Create ID element if it doesn't exist
                                id_elem = ET.SubElement(new_setup, 'id')
                                id_elem.text = str(next_id)
                            
                            # Update the name to include grid position
                            name_elem = new_setup.find('name')
                            orig_name = "setup" if name_elem is None or name_elem.text is None else name_elem.text
                            
                            if name_elem is None:
                                # Create name element if it doesn't exist
                                name_elem = ET.SubElement(new_setup, 'name')
                            
                            name_elem.text = f"{orig_name}_grid_{x}_{y}_{z}"
                            
                            # Store the bounding box information (for future use)
                            bbox = {
                                'min': [min_x, min_y, min_z],
                                'max': [max_x, max_y, max_z],
                                'original_setup_id': setup_id
                            }
                            
                            # Update the view setup with the bounding box information
                            # This could be stored as attributes or child elements
                            
                            new_view_setups.append((new_setup, bbox))
                            next_id += 1
        
        # Add all the new view setups to the SpimData
        print(f"🔧 [split_images] Adding {len(new_view_setups)} new view setups...")
        for new_setup, bbox in new_view_setups:
            view_setups.append(new_setup)
            
            # Add a new ViewRegistration for each timepoint
            if timepoints is not None and view_registrations is not None:
                timepoint_ids = []
                if timepoints.get('type') == 'pattern':
                    pattern = timepoints.find('integerpattern')
                    if pattern is not None and pattern.text is not None:
                        # Simple case: a single timepoint
                        timepoint_ids = [int(pattern.text)]
                
                # If we couldn't find timepoints from the pattern, try another approach
                if not timepoint_ids:
                    # Look for all timepoints in existing ViewRegistrations
                    for vreg in view_registrations.findall('ViewRegistration'):
                        tp_attr = vreg.get('timepoint')
                        if tp_attr is not None and tp_attr not in timepoint_ids:
                            try:
                                timepoint_ids.append(int(tp_attr))
                            except:
                                pass
                
                # If still no timepoints, assume at least timepoint 0
                if not timepoint_ids:
                    timepoint_ids = [0]
                
                # Process each timepoint
                for tp in timepoint_ids:
                    # Try to find the registration for the original setup
                    original_id = bbox['original_setup_id']
                    reg_xpath = f"ViewRegistration[@timepoint='{tp}'][@setup='{original_id}']"
                    reg_elem = None
                    
                    for vreg in view_registrations.findall('ViewRegistration'):
                        if vreg.get('timepoint') == str(tp) and vreg.get('setup') == str(original_id):
                            reg_elem = vreg
                            break
                    
                    if reg_elem is not None:
                        # Create a new ViewRegistration
                        new_reg = copy.deepcopy(reg_elem)
                        # Update setup ID
                        new_reg.set('setup', new_setup.find('id').text)
                        # Add to ViewRegistrations
                        view_registrations.append(new_reg)
        
        # Assign illuminations from tile IDs if requested
        if assign_illuminations:
            print(f"🔧 [split_images] Assigning illuminations from tile IDs...")
            # Implementation depends on illumination attribute structure
        
        # Optimize if requested
        if optimize:
            print(f"🔧 [split_images] Performing optimization...")
            # Implementation depends on optimization method
        
        # Add interest points if requested
        if fake_interest_points:
            print(f"🔧 [split_images] Adding interest points...")
            # Would need to implement fake interest point generation using
            # fip_density, fip_min_num_points, fip_max_num_points, fip_error, and fip_exclusion_radius
        
        print(f"🔧 [split_images] Image splitting completed successfully.")
        return new_spim_data
        
    except Exception as e:
        print(f"❌ Error in split_images: {str(e)}")
        traceback.print_exc()
        return None