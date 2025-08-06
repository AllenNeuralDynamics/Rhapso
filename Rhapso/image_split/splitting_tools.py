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
    import copy
    import numpy as np
    from xml.etree import ElementTree as ET

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
        
        if img_loader is not None:
            # Clone the existing loader to become the inner duplicate
            inner = copy.deepcopy(img_loader)
            # Create a new outer wrapper with the same tag & format
            outer = ET.Element(img_loader.tag, {'format': img_loader.get('format')})
            outer.append(inner)
            # Replace the original loader with our two-level one
            idx = list(seq_desc).index(img_loader)
            seq_desc.remove(img_loader)
            seq_desc.insert(idx, outer)

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
        new2oldSetupId = {}
        newSetupId2Interval = {}
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
                            # --- Add to SetupIds mapping ---
                            new2oldSetupId[next_id] = setup_id
                            newSetupId2Interval[next_id] = {
                                "min": [min_x, min_y, min_z],
                                "max": [max_x, max_y, max_z]
                            }
                            next_id += 1
        
        # --- Collect new view setups and mappings ---
        new2oldSetupId = {}
        newSetupId2Interval = {}
        new_view_setups = []
        next_id = 0

        # Get timepoints as a list of ints
        timepoints_elem = seq_desc.find('Timepoints')
        timepoints = []
        if timepoints_elem is not None:
            pattern = timepoints_elem.find('integerpattern')
            if pattern is not None and pattern.text is not None:
                timepoints = [int(x) for x in pattern.text.strip().split()]
        if not timepoints:
            timepoints = [0]

        # For each original setup, split and collect info
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
                            # --- Add to SetupIds mapping ---
                            new2oldSetupId[next_id] = setup_id
                            newSetupId2Interval[next_id] = {
                                "min": [min_x, min_y, min_z],
                                "max": [max_x, max_y, max_z]
                            }
                            next_id += 1

        # --- Build <SequenceDescription> ---
        seq_desc_elem = ET.Element("SequenceDescription")
        # <ViewSetups>
        view_setups_elem = ET.SubElement(seq_desc_elem, "ViewSetups")
        for new_setup, bbox in new_view_setups:
            # Extract info from new_setup and bbox as needed
            vs_elem = ET.SubElement(view_setups_elem, "ViewSetup")
            # id
            id_elem = new_setup.find('id')
            ET.SubElement(vs_elem, "id").text = id_elem.text if id_elem is not None else ""
            # size
            size_elem = new_setup.find('size')
            if size_elem is not None and size_elem.text:
                ET.SubElement(vs_elem, "size").text = size_elem.text
            else:
                # fallback: use bbox max-min+1
                size_str = " ".join(str(bbox['max'][i] - bbox['min'][i] + 1) for i in range(3))
                ET.SubElement(vs_elem, "size").text = size_str
            # voxelSize
            voxel_elem = new_setup.find('voxelSize')
            if voxel_elem is not None:
                voxel_xml = ET.SubElement(vs_elem, "voxelSize")
                unit_elem = voxel_elem.find('unit')
                size_elem = voxel_elem.find('size')
                ET.SubElement(voxel_xml, "unit").text = unit_elem.text if unit_elem is not None else "µm"
                ET.SubElement(voxel_xml, "size").text = size_elem.text if size_elem is not None else "1.0 1.0 1.0"
            # attributes
            attrs_elem = new_setup.find('attributes')
            if attrs_elem is not None:
                attrs_xml = ET.SubElement(vs_elem, "attributes")
                for attr in ["illumination", "channel", "tile", "angle"]:
                    val_elem = attrs_elem.find(attr)
                    ET.SubElement(attrs_xml, attr).text = val_elem.text if val_elem is not None else "0"
        # <Attributes> blocks
        # Illumination
        illum_ids = set()
        channel_ids = set()
        tile_ids = set()
        tile_locations = {}
        angle_ids = set()
        for new_setup, bbox in new_view_setups:
            attrs_elem = new_setup.find('attributes')
            if attrs_elem is not None:
                illum_elem = attrs_elem.find('illumination')
                channel_elem = attrs_elem.find('channel')
                tile_elem = attrs_elem.find('tile')
                angle_elem = attrs_elem.find('angle')
                if illum_elem is not None and illum_elem.text is not None:
                    illum_ids.add(illum_elem.text)
                if channel_elem is not None and channel_elem.text is not None:
                    channel_ids.add(channel_elem.text)
                if tile_elem is not None and tile_elem.text is not None:
                    tile_ids.add(tile_elem.text)
                    # Try to get tile location from bbox or elsewhere
                    tile_locations[tile_elem.text] = " ".join(str(x) for x in bbox['min'])
                if angle_elem is not None and angle_elem.text is not None:
                    angle_ids.add(angle_elem.text)

        illum_elem = ET.SubElement(view_setups_elem, "Attributes", name="illumination")
        for illum_id in sorted(illum_ids, key=lambda x: int(x)):
            ill = ET.SubElement(illum_elem, "Illumination")
            ET.SubElement(ill, "id").text = str(illum_id)
            ET.SubElement(ill, "name").text = str(illum_id)
        # Channel
        chan_elem = ET.SubElement(view_setups_elem, "Attributes", name="channel")
        for chan_id in sorted(channel_ids, key=lambda x: int(x)):
            ch = ET.SubElement(chan_elem, "Channel")
            ET.SubElement(ch, "id").text = str(chan_id)
            ET.SubElement(ch, "name").text = str(chan_id)
        # Tile
        tile_elem = ET.SubElement(view_setups_elem, "Attributes", name="tile")
        for tile_id in sorted(tile_ids, key=lambda x: int(x)):
            t = ET.SubElement(tile_elem, "Tile")
            ET.SubElement(t, "id").text = str(tile_id)
            ET.SubElement(t, "name").text = str(tile_id)
            ET.SubElement(t, "location").text = tile_locations.get(tile_id, "0 0 0")
        # Angle
        angle_elem = ET.SubElement(view_setups_elem, "Attributes", name="angle")
        for angle_id in sorted(angle_ids, key=lambda x: int(x)):
            ang = ET.SubElement(angle_elem, "Angle")
            ET.SubElement(ang, "id").text = str(angle_id)
            ET.SubElement(ang, "name").text = str(angle_id)
        # <Timepoints>
        timepoints_elem = ET.SubElement(seq_desc_elem, "Timepoints", type="pattern")
        ET.SubElement(timepoints_elem, "integerpattern").text = " ".join(map(str, timepoints))
        # <MissingViews /> as self-closing
        ET.SubElement(seq_desc_elem, "MissingViews")

        # --- Build <SetupIds> ---
        setup_ids_elem = ET.Element("SetupIds")
        for new_id in sorted(new2oldSetupId.keys()):
            setup_def = ET.SubElement(setup_ids_elem, "SetupIdDefinition")
            ET.SubElement(setup_def, "NewId").text = str(new_id)
            ET.SubElement(setup_def, "OldId").text = str(new2oldSetupId[new_id])
            interval = newSetupId2Interval[new_id]
            ET.SubElement(setup_def, "min").text = " ".join(str(x) for x in interval["min"])
            ET.SubElement(setup_def, "max").text = " ".join(str(x) for x in interval["max"])

        # --- Insert into XML tree ---
        # Find the outer ImageLoader (the one wrapping the inner)
        seq_desc = xml_tree.find('SequenceDescription')
        img_loader_outer = seq_desc.find('ImageLoader')
        if img_loader_outer is not None:
            # Insert SequenceDescription and SetupIds after the nested ImageLoader
            # Remove any existing SequenceDescription/SetupIds under ImageLoader if present
            for tag in ["SequenceDescription", "SetupIds"]:
                for elem in img_loader_outer.findall(tag):
                    img_loader_outer.remove(elem)
            img_loader_outer.append(seq_desc_elem)
            img_loader_outer.append(setup_ids_elem)

        # --- Add empty elements at the end of root as self-closing ---
        for tag in ["BoundingBoxes", "PointSpreadFunctions", "StitchingResults", "IntensityAdjustments"]:
            if xml_tree.find(tag) is None:
                ET.SubElement(xml_tree, tag)

        print(f"🔧 [split_images] Image splitting completed successfully.")
        return xml_tree
        
    except Exception as e:
        print(f"❌ Error in split_images: {str(e)}")
        traceback.print_exc()
        return None