import xml.etree.ElementTree as ET
import re

def extract_grid_position_from_name(name):

    match = re.search(r'Tile_X_(\d+)_Y_(\d+)', name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def generate_settings_file(dataset_xml, output_xml, existing_settings=None):
    GREEN = "-16711936"  # 0xFF00FF00 in signed int
    RED = "-65536"       # 0xFFFF0000 in signed int
    
    print(f"Reading dataset XML {dataset_xml}")
    tree = ET.parse(dataset_xml)
    root = tree.getroot()
    
    # Find all ViewSetups
    view_setups = root.findall(".//ViewSetup")
    
    if not view_setups:
        print("ERROR: No ViewSetup elements found")
        return
    
    tile_data = []
    print(len(view_setups), "ViewSetups found:")
    for view_setup in view_setups:
        # print(dict(view_setup))
        # print(view_setup, view_setup.get('id'), view_setup.get('name'))
        raw_id = view_setup.find('id').text
        print(f"DEBUG: Raw id attribute = '{raw_id}'")
        setup_id = int(raw_id)
        name_elem = view_setup.find('name')
        
        if name_elem is not None:
            name = name_elem.text
            grid_pos = extract_grid_position_from_name(name)
            
            if grid_pos:
                x_pos, y_pos = grid_pos
                tile_data.append((setup_id, x_pos, y_pos, name))
                print(f"  Setup {setup_id}: X={x_pos}, Y={y_pos} ({name})")
            else:
                print(f"  WARNING: Could not parse position from name: {name}")
        else:
            print(f"  WARNING: No name element for setup {setup_id}")
    
    if not tile_data:
        print("ERROR: No tile positions found in names")
        return
    
    print(f"\nDetected {len(tile_data)} tiles")
    
    # Read existing settings if provided to preserve min/max values
    min_val, max_val = "0.0", "30.0"
    if existing_settings:
        try:
            existing_tree = ET.parse(existing_settings)
            first_setup = existing_tree.find(".//ConverterSetup")
            if first_setup is not None:
                min_elem = first_setup.find("min")
                max_elem = first_setup.find("max")
                if min_elem is not None:
                    min_val = min_elem.text
                if max_elem is not None:
                    max_val = max_elem.text
        except:
            pass
    
    root = ET.Element("Settings")
    
    viewer_state = ET.SubElement(root, "ViewerState")
    sources = ET.SubElement(viewer_state, "Sources")
    
    for i in range(len(tile_data)):
        source = ET.SubElement(sources, "Source")
        active = ET.SubElement(source, "active")
        active.text = "true"
    
    source_groups = ET.SubElement(viewer_state, "SourceGroups")
    for i in range(min(10, len(tile_data))):
        group = ET.SubElement(source_groups, "SourceGroup")
        active = ET.SubElement(group, "active")
        active.text = "true"
        name = ET.SubElement(group, "name")
        name.text = f"group {i+1}"
        group_id = ET.SubElement(group, "id")
        group_id.text = str(tile_data[i][0])
    
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
    
    setup_assignments = ET.SubElement(root, "SetupAssignments")
    converter_setups = ET.SubElement(setup_assignments, "ConverterSetups")
    
    print("\nApplying checkerboard pattern:")
    
    tile_data_sorted = sorted(tile_data, key=lambda x: x[0])
    
    for setup_id, x_pos, y_pos, name in tile_data_sorted:
        is_green = (x_pos + y_pos) % 2 == 0
        color_value = GREEN if is_green else RED
        
        setup = ET.SubElement(converter_setups, "ConverterSetup")
        
        id_elem = ET.SubElement(setup, "id")
        id_elem.text = str(setup_id)  # This should use the actual setup_id variable
        
        min_elem = ET.SubElement(setup, "min")
        min_elem.text = min_val
        
        max_elem = ET.SubElement(setup, "max")
        max_elem.text = max_val
        
        color_elem = ET.SubElement(setup, "color")
        color_elem.text = color_value
        
        group_id = ET.SubElement(setup, "groupId")
        group_id.text = "0"
        
        color_name = "GREEN" if is_green else "RED"
        print(f"Setup {setup_id:3d} at X={x_pos}, Y={y_pos} -> {color_name}")
    
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
    
    # ManualSourceTransforms
    transforms = ET.SubElement(root, "ManualSourceTransforms")
    for i in range(len(tile_data)):
        transform = ET.SubElement(transforms, "SourceTransform")
        transform.set("type", "affine")
        affine = ET.SubElement(transform, "affine")
        affine.text = "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0"
    
    ET.SubElement(root, "Bookmarks")
    
    tree = ET.ElementTree(root)
    tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    
    print("\n" + "=" * 60)
    print(f"Settings file created: {output_xml}")
    print(f"Total tiles: {len(tile_data)}")
    print("\nColors applied based on X + Y grid position")

if __name__ == "__main__":

    vis_grid = True
    input_xml = "/Users/sean.fite/Desktop/rhapso-solver-affine.xml"
    output_xml = "/Users/sean.fite/Desktop/bdv_settings.xml"
    
    generate_settings_file(input_xml, output_xml)

# python -m Rhapso.eval.bdv_checkerboard_tile_colors
