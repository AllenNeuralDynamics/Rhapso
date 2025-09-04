"""
Image splitting dataset processor for Rhapso.

This module provides functionality to split large images into smaller tiles
while maintaining metadata and creating fake interest points for alignment.

The main function accepts configuration parameters for input/output paths,
image splitting parameters, and fake interest point settings.

Example usage:
    from Rhapso.image_split.split_datasets import main
    
    main(
        xml_input="path/to/input.xml",
        xml_output="path/to/output.xml",
        target_image_size_string="7000,7000,4000",
        target_overlap_string="128,128,128",
        fake_interest_points=True
    )
"""

import sys
import numpy as np
from xml.etree import ElementTree as ET
import zarr
import os
import s3fs
import boto3
from io import BytesIO

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
    """Load and parse XML input file from local path or S3."""
    print(f"Loading XML: {xml_input}")
    
    def load_operation():
        if xml_input.startswith('s3://'):
            # Handle S3 path
            s3 = boto3.client('s3')
            # Parse S3 URL to get bucket and key
            s3_parts = xml_input.replace('s3://', '').split('/', 1)
            bucket_name = s3_parts[0]
            key = s3_parts[1] if len(s3_parts) > 1 else ''
            
            response = s3.get_object(Bucket=bucket_name, Key=key)
            xml_string = response['Body'].read().decode('utf-8')
            tree = ET.parse(BytesIO(xml_string.encode('utf-8')))
            return tree.getroot()
        else:
            # Handle local path
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
    """Save the processed data to XML output file (local or S3)."""
    try:
        new_tree = ET.ElementTree(new_data)
        print(f"Saving new XML to: {xml_output}")
        
        if xml_output.startswith('s3://'):
            # Handle S3 output
            s3 = boto3.client('s3')
            # Parse S3 URL to get bucket and key
            s3_parts = xml_output.replace('s3://', '').split('/', 1)
            bucket_name = s3_parts[0]
            key = s3_parts[1] if len(s3_parts) > 1 else ''
            
            # Convert XML to bytes and upload to S3
            xml_bytes = BytesIO()
            new_tree.write(xml_bytes, encoding='utf-8', xml_declaration=True)
            xml_bytes.seek(0)
            s3.upload_fileobj(xml_bytes, bucket_name, key)
            print("XML saved to S3 successfully")
        else:
            # Handle local output
            format_xml_output(new_tree, xml_output)
            print("XML saved to local file successfully") 
            
    except PermissionError as e:
        print(f"Error: Permission denied when saving output file. {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error: IO error when saving output file. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to save output XML. {e}", file=sys.stderr)
        sys.exit(1)


def create_n5_files_for_fake_interest_points(xml_data, n5_output_path):
    """Create N5 files for fake interest points created during image splitting."""
    try:
        # Validate n5_output_path
        if not n5_output_path or not n5_output_path.strip():
            raise ValueError("n5_output_path is empty or not provided")
        
        print("Saving interest points multi-threaded ...")
        
        # Find all ViewInterestPointsFile elements to get all labels and setups
        vip_files = xml_data.findall('.//ViewInterestPointsFile') or xml_data.findall('.//{*}ViewInterestPointsFile')
        
        if not vip_files:
            print("No ViewInterestPointsFile elements found in XML")
            return
        
        # Determine if this is an S3 path or local path
        is_s3 = n5_output_path.startswith('s3://')
        
        if is_s3:
            # Handle S3 path - use zarr with s3fs
            s3_fs = s3fs.S3FileSystem()
            store = s3fs.S3Map(root=n5_output_path, s3=s3_fs, check=False)
            root = zarr.group(store=store, overwrite=False)
        else:
            # Handle local path - use zarr N5Store
            # Ensure the directory exists
            os.makedirs(os.path.dirname(n5_output_path), exist_ok=True)
            store = zarr.N5Store(n5_output_path)
            root = zarr.group(store=store, overwrite=False)
        
        # Use a different attribute name to avoid N5 reserved keyword warning
        root.attrs['n5_version'] = '4.0.0'
        
        for vip_file in vip_files:
            timepoint_attr = vip_file.get('timepoint', '0')
            setup_attr = vip_file.get('setup', '0')
            label_attr = vip_file.get('label', 'beads')
            
            # Create N5 path for this label
            n5_dataset_path = f"tpId_{timepoint_attr}_viewSetupId_{setup_attr}/{label_attr}/interestpoints"
            
            # Create empty N5 directories for fake interest points
            if n5_dataset_path not in root:
                try:
                    dataset = root.create_group(n5_dataset_path)
                except zarr.errors.ContainsGroupError:
                    # If group already exists, get it
                    dataset = root[n5_dataset_path]
                
                # Set attributes
                dataset.attrs["pointcloud"] = "1.0.0"
                dataset.attrs["type"] = "list"
                dataset.attrs["list version"] = "1.0.0"
                
                # Create sub-datasets
                id_dataset = f"{n5_dataset_path}/id"
                loc_dataset = f"{n5_dataset_path}/loc"
                intensities_dataset = f"{n5_dataset_path}/intensities" 
                
                # Create empty datasets for fake interest points
                if id_dataset not in root:
                    root.create_dataset(
                        id_dataset,
                        shape=(0,), 
                        dtype='u8',  
                        chunks=(1,),  
                        compressor=zarr.GZip()
                    )
                
                if loc_dataset not in root:
                    root.create_dataset(
                        loc_dataset,
                        shape=(0,),  
                        dtype='f8',  
                        chunks=(1,), 
                        compressor=zarr.GZip()
                    )
                
                if intensities_dataset not in root:
                    root.create_dataset(
                        intensities_dataset,
                        shape=(0,), 
                        dtype='f4', 
                        chunks=(1,),  
                        compressor=zarr.GZip()
                    )
                
                # Log the creation of fake interest point directories
                saved_path = f"file:{n5_output_path}/{n5_dataset_path}"
                print(f"Saved: {saved_path}")
        
        print("Fake interest points N5 files created successfully")
        
    except Exception as e:
        print(f"Warning: Could not create N5 files for fake interest points: {e}")
        print("Continuing without N5 file creation...")


# ============================================================================
# Main Function
# ============================================================================

def main(
    xml_input,
    xml_output=None,
    n5_output=None,
    target_image_size_string=None,
    target_overlap_string=None,
    fake_interest_points=False,
    fip_exclusion_radius=20.0,
    assign_illuminations=False,
    disable_optimization=False,
    fip_density=100.0,
    fip_min_num_points=20,
    fip_max_num_points=500,
    fip_error=0.5
):
    """Main entry point for image splitting process."""
    print("beginning image splitting...")
    
    # Validate required parameters
    if not xml_input:
        raise ValueError("xml_input is required and cannot be empty")
    if not target_image_size_string:
        raise ValueError("target_image_size_string is required and cannot be empty")
    if not target_overlap_string:
        raise ValueError("target_overlap_string is required and cannot be empty")
    
    # If xml_output is not provided, default to overwriting input XML
    if xml_output is None:
        xml_output = xml_input
        print(f"Warning: xml_output not provided, will overwrite input XML: {xml_input}")
    
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
    
    # Step 8: Create N5 files for fake interest points if enabled
    if fake_interest_points and n5_output:
        create_n5_files_for_fake_interest_points(new_data, n5_output)
    elif fake_interest_points and not n5_output:
        print("Warning: fake_interest_points is True but n5_output is not provided. Skipping N5 file creation.")

    print("Split-Images run finished")


if __name__ == '__main__':
    # Example usage when run directly
    # This shows how to call the main function with parameters
    print("Example: This module is designed to be imported and called with parameters.")
    print("Use the image_split_pipeline.py script or import this module directly.")
    print("Example usage:")
    print("  from Rhapso.image_split.split_datasets import main")
    print("  main(xml_input='path/to/input.xml', target_image_size_string='7000,7000,4000', target_overlap_string='128,128,128')")