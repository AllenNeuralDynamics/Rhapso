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

# Ray imports (optional)
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


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
        
        # First, create correspondences folders for ALL labels (including original labels)
        print("Creating correspondences folders for all labels...")
        correspondences_created = 0
        for vip_file in vip_files:
            timepoint_attr = vip_file.get('timepoint', '0')
            setup_attr = vip_file.get('setup', '0')
            label_attr = vip_file.get('label', 'beads')
            
            # Create N5 path for this label (correspondences at same level as interestpoints)
            n5_dataset_path = f"tpId_{timepoint_attr}_viewSetupId_{setup_attr}/{label_attr}"
            correspondences_path = f"{n5_dataset_path}/correspondences"
            
            if correspondences_path not in root:
                try:
                    correspondences_group = root.create_group(correspondences_path)
                    # Set correspondences attributes
                    correspondences_group.attrs["correspondences"] = "1.0.0"
                    correspondences_group.attrs["idMap"] = {}  # Empty idMap initially
                    correspondences_created += 1
                except Exception as e:
                    print(f"Warning: Could not create correspondences folder {correspondences_path}: {e}")
        
        print(f"Created {correspondences_created} correspondences folders")
        
        # Then, create fake interest points for split labels only
        print("Creating fake interest points for split labels...")
        skipped_original = 0
        fake_points_created = 0
        for vip_file in vip_files:
            timepoint_attr = vip_file.get('timepoint', '0')
            setup_attr = vip_file.get('setup', '0')
            label_attr = vip_file.get('label', 'beads')
            
            # Only create fake interest points for split labels
            if not (label_attr.endswith('_split') or label_attr.startswith('splitPoints_')):
                skipped_original += 1
                continue
            
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
                fake_points_created += 1
        
        if skipped_original > 0:
            print(f"Skipped {skipped_original} original labels (no fake interest points needed)")
        print(f"Created {fake_points_created} fake interest point datasets")
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


# ============================================================================ 
# Ray Distribution Functions
# ============================================================================

@ray.remote
def create_correspondences_folder(correspondences_data, n5_output_path, is_s3=False):
    """
    Create a correspondences folder for a single label.
    This function is designed to be used as a Ray remote task.
    
    Args:
        correspondences_data: Dictionary containing timepoint, setup, and label info
        n5_output_path: Path to the N5 output file
        is_s3: Whether the output is to S3 storage
        
    Returns:
        Dictionary with success status and path
    """
    try:
        timepoint = correspondences_data['timepoint']
        setup = correspondences_data['setup']
        label = correspondences_data['label']
        
        # Create N5 path for this label (correspondences at same level as interestpoints)
        n5_dataset_path = f"tpId_{timepoint}_viewSetupId_{setup}/{label}"
        correspondences_path = f"{n5_dataset_path}/correspondences"
        
        if is_s3:
            # Handle S3 path - use zarr with s3fs
            s3_fs = s3fs.S3FileSystem()
            store = s3fs.S3Map(root=n5_output_path, s3=s3_fs, check=False)
            root = zarr.group(store=store, overwrite=False)
        else:
            # Handle local path - use zarr with N5Store
            os.makedirs(os.path.dirname(n5_output_path), exist_ok=True)
            store = zarr.N5Store(n5_output_path)
            root = zarr.group(store=store, overwrite=False)
        
        # Create correspondences folder
        if correspondences_path not in root:
            correspondences_group = root.create_group(correspondences_path)
            # Set correspondences attributes
            correspondences_group.attrs["correspondences"] = "1.0.0"
            correspondences_group.attrs["idMap"] = {}  # Empty idMap initially
        
        return {
            "success": True, 
            "path": correspondences_path
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "path": correspondences_path}


@ray.remote
def create_single_n5_dataset(vip_file_data, n5_output_path, is_s3=False, fip_density=100.0, fip_min_num_points=20, fip_max_num_points=500):
    """
    Create a single N5 dataset for fake interest points.
    This function is designed to be used as a Ray remote task.
    
    Args:
        vip_file_data: Dictionary containing timepoint, setup, and label info
        n5_output_path: Path to the N5 output file
        is_s3: Whether the output is to S3 storage
        fip_density: Density of fake interest points per unit volume
        fip_min_num_points: Minimum number of fake points to generate
        fip_max_num_points: Maximum number of fake points to generate
        
    Returns:
        Dictionary with success status, path, and metadata
    """
    try:
        timepoint_attr = vip_file_data['timepoint']
        setup_attr = vip_file_data['setup']
        label_attr = vip_file_data['label']
        
        # Create N5 path for this label
        n5_dataset_path = f"tpId_{timepoint_attr}_viewSetupId_{setup_attr}/{label_attr}/interestpoints"
        
        if is_s3:
            # Handle S3 path - use zarr with s3fs
            s3_fs = s3fs.S3FileSystem()
            store = s3fs.S3Map(root=n5_output_path, s3=s3_fs, check=False)
            root = zarr.group(store=store, overwrite=False)
        else:
            # Handle local path - use zarr with N5Store
            # Ensure the directory exists
            os.makedirs(os.path.dirname(n5_output_path), exist_ok=True)
            store = zarr.N5Store(n5_output_path)
            root = zarr.group(store=store, overwrite=False)
        
        # Create N5 group for fake interest points
        try:
            if n5_dataset_path not in root:
                dataset = root.create_group(n5_dataset_path)
            else:
                dataset = root[n5_dataset_path]
            
            # Set attributes
            dataset.attrs["pointcloud"] = "1.0.0"
            dataset.attrs["type"] = "list"
            dataset.attrs["list version"] = "1.0.0"
                
        except Exception as e:
            # If there's a race condition or other error, try to continue
            print(f"Warning: Could not create group {n5_dataset_path}: {e}")
            return {"success": False, "error": f"Group creation failed: {e}", "dataset_path": n5_dataset_path}
        
        # Generate fake interest points
        # Use setup ID as seed for reproducible fake points
        np.random.seed(int(setup_attr) + 42)  # Add offset to avoid identical points across setups
        
        # Calculate grid size based on density and target size
        # Use a reasonable default target size for fake points
        target_size = [1000, 1000, 1000]  # Default target size in voxels
        grid_size = int((fip_density * np.prod(target_size)) ** (1/3))
        grid_size = max(2, min(grid_size, 50))  # Reasonable bounds
        
        # Generate grid of points
        x_coords = np.linspace(0, target_size[0], grid_size)
        y_coords = np.linspace(0, target_size[1], grid_size)
        z_coords = np.linspace(0, target_size[2], grid_size)
        
        # Create meshgrid
        xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        # Add some random noise to make it more realistic
        noise_scale = min(target_size) * 0.1  # 10% of smallest dimension
        noise = np.random.normal(0, noise_scale, grid_points.shape)
        grid_points += noise
        
        # Randomly sample points to get desired count
        num_points = min(len(grid_points), fip_max_num_points)
        num_points = max(num_points, fip_min_num_points)
        
        if len(grid_points) > num_points:
            indices = np.random.choice(len(grid_points), num_points, replace=False)
            fake_points = grid_points[indices]
        else:
            fake_points = grid_points
        
        # Generate fake intensities (random values between 0.1 and 1.0)
        fake_intensities = np.random.uniform(0.1, 1.0, num_points)
        
        # Create fake IDs (sequential) - must be reshaped to (num_points, 1)
        fake_ids = np.arange(num_points, dtype=np.uint64).reshape(-1, 1)
        
        # Create sub-datasets with actual data
        id_dataset = f"{n5_dataset_path}/id"
        loc_dataset = f"{n5_dataset_path}/loc"
        intensities_dataset = f"{n5_dataset_path}/intensities"
        
        # Remove existing datasets if they exist (with error handling)
        try:
            if id_dataset in root:
                del root[id_dataset]
            if loc_dataset in root:
                del root[loc_dataset]
            if intensities_dataset in root:
                del root[intensities_dataset]
        except Exception as e:
            print(f"Warning: Could not delete existing datasets: {e}")
        
        # Create the datasets with actual data
        try:
            # Create ID dataset (2D array: num_points x 1)
            root.create_dataset(
                id_dataset,
                data=fake_ids,
                dtype='u8',
                chunks=(min(100, num_points), 1),  # Corrected chunking for 2D array
                compressor=zarr.GZip()
            )
            
            # Create location dataset (2D array: num_points x 3)
            root.create_dataset(
                loc_dataset,
                data=fake_points,
                dtype='f8',
                chunks=(min(100, num_points), 3),  # Corrected chunking for 2D array
                compressor=zarr.GZip()
            )
            
            # Create intensities dataset (1D array: num_points)
            intensities_ds = root.create_dataset(
                intensities_dataset,
                data=fake_intensities,
                dtype='f4',
                chunks=(min(100, num_points),),
                compressor=zarr.GZip()
            )
            # Add required attributes for intensities dataset (use non-reserved names)
            intensities_ds.attrs["dataset_dimensions"] = [1, num_points]
            intensities_ds.attrs["dataset_blockSize"] = [1, min(100, num_points)]
            
        except Exception as e:
            print(f"Warning: Could not create datasets for {n5_dataset_path}: {e}")
            return {"success": False, "error": f"Dataset creation failed: {e}", "dataset_path": n5_dataset_path}
        
        saved_path = f"file:{n5_output_path}/{n5_dataset_path}"
        return {
            "success": True, 
            "path": saved_path, 
            "dataset_path": n5_dataset_path,
            "num_points": num_points
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "dataset_path": n5_dataset_path}


def create_n5_files_for_fake_interest_points_ray(xml_data, n5_output_path, fip_density=100.0, fip_min_num_points=20, fip_max_num_points=500):
    """
    Create N5 files for fake interest points using Ray for parallel processing.
    
    Args:
        xml_data: Parsed XML data containing view information
        n5_output_path: Path to the N5 output file
        fip_density: Density of fake interest points per unit volume
        fip_min_num_points: Minimum number of fake points to generate
        fip_max_num_points: Maximum number of fake points to generate
    """
    try:
        if not RAY_AVAILABLE:
            print("Ray is not available. Falling back to sequential processing.")
            return create_n5_files_for_fake_interest_points(xml_data, n5_output_path)
        
        # Find all ViewInterestPointsFile elements
        vip_files = []
        for vip_file in xml_data.findall('.//ViewInterestPointsFile'):
            vip_files.append(vip_file.attrib)
        
        if not vip_files:
            print("No ViewInterestPointsFile elements found in XML data.")
            return
        
        # Determine if this is an S3 path
        is_s3 = n5_output_path.startswith('s3://')
        
        # Create the root N5 structure first (before parallel processing)
        print("Creating root N5 structure...")
        if is_s3:
            # Handle S3 path - use zarr with s3fs
            s3_fs = s3fs.S3FileSystem()
            store = s3fs.S3Map(root=n5_output_path, s3=s3_fs, check=False)
            root = zarr.group(store=store, overwrite=False)
        else:
            # Handle local path - use zarr with N5Store
            # Ensure the directory exists
            os.makedirs(os.path.dirname(n5_output_path), exist_ok=True)
            store = zarr.N5Store(n5_output_path)
            root = zarr.group(store=store, overwrite=False)
        
        # Set root attributes
        root.attrs['n5_version'] = '4.0.0'  # Non-reserved name
        
        print(f"Found {len(vip_files)} ViewInterestPointsFile elements to process")
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=4)  # Use 4 CPUs to reduce file system contention
        
        # First, create correspondences folders for ALL labels using Ray for speed
        print("Creating correspondences folders for all labels using Ray...")
        
        # Prepare data for parallel correspondences folder creation
        correspondences_data_list = []
        for vip_file in vip_files:
            timepoint = vip_file.get('timepoint', '0')
            setup = vip_file.get('setup', '0')
            label = vip_file.get('label', 'beads')
            
            correspondences_data = {
                'timepoint': timepoint,
                'setup': setup,
                'label': label
            }
            correspondences_data_list.append(correspondences_data)
        
        # Create Ray remote tasks for correspondences folder creation
        correspondences_futures = []
        for correspondences_data in correspondences_data_list:
            future = create_correspondences_folder.remote(correspondences_data, n5_output_path, is_s3)
            correspondences_futures.append(future)
        
        # Collect results
        correspondences_results = ray.get(correspondences_futures)
        
        # Count successful correspondences folder creation
        successful_correspondences = sum(1 for result in correspondences_results if result['success'])
        print(f"Created {successful_correspondences}/{len(correspondences_results)} correspondences folders")
        
        # Then, prepare data for parallel processing of fake interest points
        # Only create fake interest points for split labels, not original labels
        vip_file_data_list = []
        skipped_original = 0
        for vip_file in vip_files:
            label = vip_file.get('label', 'beads')
            # Only process labels that end with "_split" or start with "splitPoints_"
            if label.endswith('_split') or label.startswith('splitPoints_'):
                vip_file_data = {
                    'timepoint': vip_file.get('timepoint', '0'),
                    'setup': vip_file.get('setup', '0'),
                    'label': label
                }
                vip_file_data_list.append(vip_file_data)
            else:
                skipped_original += 1
        
        if skipped_original > 0:
            print(f"Skipped {skipped_original} original labels (no fake interest points needed)")
        
        # Create Ray remote tasks for parallel execution
        futures = []
        for vip_file_data in vip_file_data_list:
            future = create_single_n5_dataset.remote(vip_file_data, n5_output_path, is_s3, fip_density, fip_min_num_points, fip_max_num_points)
            futures.append(future)
        
        # Collect results
        print("Processing N5 datasets in parallel...")
        results = ray.get(futures)
        
        # Process results
        successful = 0
        failed = 0
        total_points = 0
        
        for result in results:
            if result['success']:
                successful += 1
                total_points += result.get('num_points', 0)
                print(f"✓ Created: {result['path']} ({result.get('num_points', 0)} points)")
            else:
                failed += 1
                print(f"✗ Failed: {result.get('dataset_path', 'unknown')} - {result.get('error', 'unknown error')}")
        
        print(f"\nN5 creation completed: {successful} successful, {failed} failed")
        print(f"Total fake interest points created: {total_points}")
        
    except Exception as e:
        print(f"Error in parallel N5 creation: {e}")
        # Fall back to sequential processing
        print("Falling back to sequential processing...")
        create_n5_files_for_fake_interest_points(xml_data, n5_output_path)


def main_with_ray(xml_input, xml_output, n5_output=None, target_image_size_string="7000,7000,4000", 
                  target_overlap_string="128,128,128", fake_interest_points=False, fip_density=100.0, 
                  fip_min_num_points=20, fip_max_num_points=500, fip_exclusion_radius=200.0,
                  assign_illuminations=False, disable_optimization=False, fip_error=0.5):
    """
    Main function with Ray distribution support for N5 creation.
    
    This function runs the image splitting pipeline and optionally creates fake interest points
    using Ray for parallel N5 file creation.
    
    Args:
        xml_input: Path to input XML file
        xml_output: Path to output XML file  
        n5_output: Path to N5 output file (optional)
        target_image_size_string: Target image size as comma-separated string
        target_overlap_string: Target overlap as comma-separated string
        fake_interest_points: Whether to create fake interest points
        fip_density: Density of fake interest points per unit volume
        fip_min_num_points: Minimum number of fake points to generate
        fip_max_num_points: Maximum number of fake points to generate
        fip_exclusion_radius: Exclusion radius for fake points
        assign_illuminations: Whether to assign illuminations
        disable_optimization: Whether to disable optimization
        fip_error: Error threshold for fake points
    """
    try:
        # Initialize Ray if available
        if RAY_AVAILABLE and not ray.is_initialized():
            ray.init(num_cpus=4)  # Use 4 CPUs to reduce file system contention
        
        # Run the main splitting pipeline
        main(
            xml_input=xml_input,
            xml_output=xml_output,
            n5_output=None,  # N5 creation handled separately with Ray
            target_image_size_string=target_image_size_string,
            target_overlap_string=target_overlap_string,
            fake_interest_points=False,  # Disable in main, handle separately
            fip_density=fip_density,
            fip_min_num_points=fip_min_num_points,
            fip_max_num_points=fip_max_num_points,
            fip_exclusion_radius=fip_exclusion_radius,
            assign_illuminations=assign_illuminations,
            disable_optimization=disable_optimization,
            fip_error=fip_error
        )
        
        # Create fake interest points with Ray if enabled
        if fake_interest_points and n5_output:
            print("Creating fake interest points with Ray distribution...")
            # Load the output XML to get the view information
            xml_data = ET.parse(xml_output)
            create_n5_files_for_fake_interest_points_ray(
                xml_data, n5_output, fip_density, fip_min_num_points, fip_max_num_points
            )
        elif fake_interest_points and not n5_output:
            print("Warning: fake_interest_points is True but n5_output is not provided. Skipping N5 file creation.")
        
        print("Image split pipeline with Ray distribution completed.")
        
    finally:
        # Shutdown Ray if it was initialized
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()


if __name__ == '__main__':
    # Example usage when run directly
    # This shows how to call the main function with parameters
    print("Example: This module is designed to be imported and called with parameters.")
    print("Use the image_split_pipeline.py script or import this module directly.")
    print("Example usage:")
    print("  from Rhapso.image_split.split_datasets import main")
    print("  main(xml_input='path/to/input.xml', target_image_size_string='7000,7000,4000', target_overlap_string='128,128,128')")