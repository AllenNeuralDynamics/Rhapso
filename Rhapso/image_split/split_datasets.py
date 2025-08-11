"""
Image splitting dataset processor for Rhapso.
python3 -m Rhapso.image_split.SplitDatasets
"""

import sys
import numpy as np
from xml.etree import ElementTree as ET

from Rhapso.image_split.split_views import collect_image_sizes, find_min_step_size, next_multiple
from Rhapso.image_split.splitting_tools import split_images


# ============================================================================
# Utility Functions
# ============================================================================

def closest_larger_long_divisable_by(value, divisor):
    """
    Find the closest larger number that is divisible by the given divisor.
    This is equivalent to the Java function closestLargerLongDivisableBy.
    
    Args:
        value: The input value to find the closest larger divisible number for
        divisor: The divisor that the result must be divisible by
        
    Returns:
        The smallest number >= value that is divisible by divisor
    """
    if divisor <= 0:
        raise ValueError("Divisor must be positive")
    
    # Calculate how many complete divisors fit in the value
    quotient = value // divisor
    
    # If value is already divisible, return it
    if value % divisor == 0:
        return value
    
    # Otherwise, return the next multiple of the divisor
    return (quotient + 1) * divisor


def safe_parse_dimensions(dim_string, name="dimensions"):
    """Safely parse comma-separated dimension string."""
    try:
        dimensions = np.array([int(s) for s in dim_string.split(',')], dtype=np.int64)
        if len(dimensions) != 3:
            raise ValueError(f"{name} must contain exactly 3 values (x,y,z)")
        return dimensions
    except ValueError as e:
        raise ValueError(f"Invalid {name}: {e}. Must be comma-separated integers.")


def safe_xml_operation(operation, error_msg, exit_on_error=True):
    """Safely execute XML operations with consistent error handling."""
    try:
        return operation()
    except ET.ParseError as e:
        print(f"Error: Could not parse XML file. {e}", file=sys.stderr)
        if exit_on_error:
            sys.exit(1)
        return None
    except FileNotFoundError as e:
        print(f"Error: XML file not found. {e}", file=sys.stderr)
        if exit_on_error:
            sys.exit(1)
        return None
    except Exception as e:
        print(f"Error: {error_msg}. {e}", file=sys.stderr)
        if exit_on_error:
            sys.exit(1)
        return None


def format_xml_output(tree, output_path):
    """Format and save XML output with consistent formatting."""
    def indent(elem, level=0):
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    # Use built-in indent for Python 3.9+, otherwise use our custom function
    if sys.version_info.major == 3 and sys.version_info.minor >= 9:
        ET.indent(tree, space="  ", level=0)
    else:
        indent(tree.getroot())
    
    # Save with consistent formatting
    tree.write(
        output_path, 
        encoding='utf-8', 
        xml_declaration=True, 
        method="xml", 
        short_empty_elements=True
    )


# ============================================================================
# Main Processing Functions
# ============================================================================

def load_xml_data(xml_input):
    """Load and parse XML input file."""
    print(f"Loading XML: {xml_input}")
    
    def load_operation():
        tree = ET.parse(xml_input)
        return tree.getroot()
    
    return safe_xml_operation(load_operation, "Unexpected error loading XML")


def process_target_parameters(target_image_size_string, target_overlap_string):
    """Parse and validate target image size and overlap parameters."""
    try:
        target_image_size = safe_parse_dimensions(target_image_size_string, "Target image size")
        target_overlap = safe_parse_dimensions(target_overlap_string, "Target overlap")
        
        print(f"Target image size: {target_image_size}")
        print(f"Target overlap: {target_overlap}")
        
        return target_image_size, target_overlap
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def analyze_dataset(data_global):
    """Analyze the dataset to collect image sizes and determine minimum step size."""
    # Collect image sizes
    img_sizes = safe_xml_operation(
        lambda: collect_image_sizes(data_global),
        "Failed to collect image sizes"
    )
    
    if img_sizes:
        print("Current image sizes of dataset:")
        for size_str, count in img_sizes[0].items():
            print(f"{count}x: {size_str}")
    
    # Find minimum step size
    min_step_size = safe_xml_operation(
        lambda: find_min_step_size(data_global),
        "Failed to find minimum step size"
    )
    
    print(f"Target image sizes and overlaps need to be divisible by {min_step_size}")
    
    return min_step_size


def calculate_adjusted_parameters(target_image_size, target_overlap, min_step_size):
    """Calculate adjusted target sizes and overlaps based on minimum step size."""
    try:
        # Adjust target image sizes
        sx = closest_larger_long_divisable_by(target_image_size[0], min_step_size[0])
        sy = closest_larger_long_divisable_by(target_image_size[1], min_step_size[1])
        sz = closest_larger_long_divisable_by(target_image_size[2], min_step_size[2])
        
        # Adjust target overlaps
        ox = closest_larger_long_divisable_by(target_overlap[0], min_step_size[0])
        oy = closest_larger_long_divisable_by(target_overlap[1], min_step_size[1])
        oz = closest_larger_long_divisable_by(target_overlap[2], min_step_size[2])

        print(f"Adjusted target image size: [{sx}, {sy}, {sz}]")
        print(f"Adjusted target overlap: [{ox}, {oy}, {oz}]")
        
        return [sx, sy, sz], [ox, oy, oz]
    except Exception as e:
        print(f"Error: Failed to calculate adjusted sizes and overlaps. {e}", file=sys.stderr)
        sys.exit(1)


def validate_parameters(adjusted_size, adjusted_overlap):
    """Validate that overlap is not bigger than size."""
    if (adjusted_overlap[0] > adjusted_size[0] or 
        adjusted_overlap[1] > adjusted_size[1] or 
        adjusted_overlap[2] > adjusted_size[2]):
        print("Error: Overlap cannot be bigger than size.", file=sys.stderr)
        sys.exit(1)


def perform_image_splitting(data_global, adjusted_overlap, adjusted_size, min_step_size, 
                           assign_illuminations, disable_optimization, fake_interest_points,
                           fip_density, fip_min_num_points, fip_max_num_points, fip_error, 
                           fip_exclusion_radius):
    """Perform the actual image splitting operation."""
    print("\nStarting image splitting process...")
    
    try:
        new_data = split_images(
            data_global,
            adjusted_overlap,
            adjusted_size,
            min_step_size,
            assign_illuminations,
            not disable_optimization,
            fake_interest_points,
            fip_density,
            fip_min_num_points,
            fip_max_num_points,
            fip_error,
            fip_exclusion_radius
        )
        
        print("Splitting process finished.")
        
        if new_data is None:
            print("Error: split_images returned None. Aborting.", file=sys.stderr)
            sys.exit(1)
        
        return new_data
        
    except KeyError as e:
        print(f"Error: XML structure missing expected elements for splitting. {e}", file=sys.stderr)
        sys.exit(1)
    except MemoryError as e:
        print(f"Error: Insufficient memory for splitting operation. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed during image splitting process. {e}", file=sys.stderr)
        sys.exit(1)


def save_xml_output(new_data, xml_output):
    """Save the processed data to XML output file."""
    try:
        new_tree = ET.ElementTree(new_data)
        print(f"Saving new XML to: {xml_output}")
        format_xml_output(new_tree, xml_output)
        print("Done.")
    except PermissionError as e:
        print(f"Error: Permission denied when saving output file. {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error: IO error when saving output file. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to save output XML. {e}", file=sys.stderr)
        sys.exit(1)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point for image splitting process."""
    print("beginning image splitting...")
    
    # Configuration - these would be replaced with proper argument parsing
    xml_input = "/mnt/c/Users/marti/Documents/allen/data/exaSPIM_686951 EXAMPLE/ip_affine_alignment/bigstitcher_affine.xml"
    xml_output = "/mnt/c/Users/marti/Documents/allen/data/exaSPIM_686951 EXAMPLE/results/bigstitcher_affine_split-RHAPSO.xml"
    target_image_size_string = "7000,7000,4000"
    target_overlap_string = "128,128,128"
    fake_interest_points = True
    fip_exclusion_radius = 200
    assign_illuminations = True
    
    # Default values
    disable_optimization = False
    fip_density = 100.0
    fip_min_num_points = 20
    fip_max_num_points = 500
    fip_error = 0.5
    
    # Register namespace for BigStitcher XML
    try:
        ET.register_namespace('', 'SpimData.xsd')
    except AttributeError as e:
        print(f"Warning: Could not register namespace: {e}", file=sys.stderr)

    # Step 1: Load XML data
    data_global = load_xml_data(xml_input)
    if data_global is None:
        return

    # Step 2: Parse target parameters
    target_image_size, target_overlap = process_target_parameters(
        target_image_size_string, target_overlap_string
    )

    # Step 3: Analyze dataset
    min_step_size = analyze_dataset(data_global)

    # Step 4: Calculate adjusted parameters
    adjusted_size, adjusted_overlap = calculate_adjusted_parameters(
        target_image_size, target_overlap, min_step_size
    )

    # Step 5: Validate parameters
    validate_parameters(adjusted_size, adjusted_overlap)

    # Step 6: Perform image splitting
    new_data = perform_image_splitting(
        data_global, adjusted_overlap, adjusted_size, min_step_size,
        assign_illuminations, disable_optimization, fake_interest_points,
        fip_density, fip_min_num_points, fip_max_num_points, fip_error,
        fip_exclusion_radius
    )

    # Step 7: Save output
    save_xml_output(new_data, xml_output)

    print("Split-Images run finished")


if __name__ == '__main__':
    main()