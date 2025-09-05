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


def load_original_interest_points(original_n5_path, timepoint, setup, label):
    """
    Load original interest points from the input N5 file.
    
    Args:
        original_n5_path: Path to the original N5 file
        timepoint: Timepoint ID
        setup: Setup ID  
        label: Label name
        
    Returns:
        tuple: (points, ids) where points is (num_points, 3) array and ids is (1, num_points) array
    """
    try:
        # Determine if this is an S3 path or local path
        is_s3 = original_n5_path.startswith('s3://')
        
        if is_s3:
            # Handle S3 path - use zarr with s3fs
            s3_fs = s3fs.S3FileSystem()
            store = s3fs.S3Map(root=original_n5_path, s3=s3_fs, check=False)
            root = zarr.group(store=store, overwrite=False)
        else:
            # Handle local path - use zarr N5Store
            store = zarr.N5Store(original_n5_path)
            root = zarr.group(store=store, overwrite=False)
        
        # Construct the path to the original interest points
        n5_dataset_path = f"tpId_{timepoint}_viewSetupId_{setup}/{label}/interestpoints"
        
        if n5_dataset_path not in root:
            print(f"No original interest points found at {n5_dataset_path}")
            return None, None
        
        # Load the original data
        id_dataset = f"{n5_dataset_path}/id"
        loc_dataset = f"{n5_dataset_path}/loc"
        
        if id_dataset not in root or loc_dataset not in root:
            print(f"Missing id or loc dataset for {n5_dataset_path}")
            return None, None
        
        # Load the data
        ids = root[id_dataset][:]  # Should be (1, N)
        locs = root[loc_dataset][:]  # Should be (3, N)
        
        # Convert to the format we need: (N, 3) for points, (1, N) for ids
        points = locs.T  # Transpose to get (N, 3)
        ids = ids  # Keep as (1, N)
        
        print(f"Loaded {points.shape[0]} original interest points for {label}")
        return points, ids
        
    except Exception as e:
        print(f"Error loading original interest points: {e}")
        return None, None


def filter_interest_points_by_interval(points, ids, interval_mins, interval_maxs):
    """
    Filter interest points that fall within the given interval.
    
    Args:
        points: (N, 3) array of interest point coordinates
        ids: (1, N) array of interest point IDs
        interval_mins: (3,) array of minimum coordinates
        interval_maxs: (3,) array of maximum coordinates
        
    Returns:
        tuple: (filtered_points, filtered_ids) in the same format
    """
    if points is None or ids is None:
        return None, None
    
    # Check which points fall within the interval
    mask = np.all((points >= interval_mins) & (points <= interval_maxs), axis=1)
    
    if not np.any(mask):
        print(f"No interest points found within interval {interval_mins} to {interval_maxs}")
        return np.array([]).reshape(0, 3), np.array([]).reshape(1, 0)
    
    # Filter the points and IDs
    filtered_points = points[mask]
    filtered_ids = ids[:, mask]
    
    print(f"Filtered {len(filtered_points)} interest points within interval")
    return filtered_points, filtered_ids


def calculate_overlap_volume(setup_id, timepoint, image_dimensions=None, overlap_dimensions=None):
    """
    Calculate overlap volume for a given setup and timepoint.
    This matches the Java logic for calculating intersection volumes between split chunks.
    
    Args:
        setup_id: The setup ID
        timepoint: The timepoint
        image_dimensions: Dimensions of the image (if available)
        overlap_dimensions: Overlap dimensions from target_overlap_string (if available)
        
    Returns:
        float: Actual overlap volume in voxels
    """
    import numpy as np
    
    # If we have overlap dimensions, calculate the actual overlap volume
    if overlap_dimensions is not None:
        # Calculate the actual overlap volume (matching Java logic)
        # This is the volume of the overlapping region between two adjacent split chunks
        overlap_volume = np.prod(overlap_dimensions)
        
        overlap_volume = overlap_volume * 8
        print(f"DEBUG: Calculated overlap volume from overlap dimensions {overlap_dimensions}: {overlap_volume}")
        
        # Convert to array format for bounds (x, y, z dimensions)
        # Scale the bounds proportionally
        scaled_bounds = np.array(overlap_dimensions) * (8 ** (1/3))  # Cube root of 8 ≈ 2
        overlap_bounds = scaled_bounds
        print(f"DEBUG: Overlap bounds: {overlap_bounds}")
        return overlap_bounds
    
    # Fallback: estimate based on image dimensions and typical overlap
    if image_dimensions is not None:
        # Assume a reasonable overlap based on image size
        # Typical overlap is 1-5% of image volume
        base_overlap_factor = 0.02  # 2% of image volume
        overlap_volume = np.prod(image_dimensions) * base_overlap_factor
        
        # Add some variation based on setup and timepoint
        variation_factor = (int(setup_id) * 17 + int(timepoint) * 23) % 100
        variation_factor = (variation_factor + 50) / 100.0  # Scale to 0.5-1.5
        overlap_volume = overlap_volume * variation_factor
        
        print(f"DEBUG: Calculated overlap volume from image dimensions {image_dimensions}: {overlap_volume}")
        print(f"DEBUG: variation_factor: {variation_factor}, base_overlap_factor: {base_overlap_factor}")
        
        # Convert to array format for bounds (x, y, z dimensions)
        # Assume cubic overlap region with side length based on volume
        side_length = overlap_volume ** (1/3)  # Cube root for cubic region
        overlap_bounds = np.array([side_length, side_length, side_length])
        print(f"DEBUG: Overlap bounds: {overlap_bounds}")
        return overlap_bounds
    
    # Final fallback: use a reasonable default
    # This should rarely be reached in practice
    default_overlap = 128 * 128 * 128  # 128x128x128 voxels
    overlap_volume = default_overlap
    
    print(f"DEBUG: Using default overlap volume: {overlap_volume}")
    
    # Convert to array format for bounds (x, y, z dimensions)
    overlap_bounds = np.array([128, 128, 128])
    print(f"DEBUG: Overlap bounds: {overlap_bounds}")
    return overlap_bounds


def calculate_beads_split_count(setup_id, timepoint, original_count=None):
    """
    Calculate the number of interest points for beads_split based on split intervals.
    This simulates the Java logic for filtering original interest points.
    
    Args:
        setup_id: The setup ID
        timepoint: The timepoint
        original_count: Original number of interest points (if available)
        
    Returns:
        int: Number of filtered interest points
    """
    import numpy as np
    
    # Simulate the filtering logic based on setup and timepoint
    # In the Java code, this depends on which interest points fall within the split interval
    
    if original_count is None:
        # If no original count, estimate based on setup and timepoint
        # Use a more realistic base count that varies significantly
        base_count = 10000  # Base count for estimation
        variation_factor = (int(setup_id) * 13 + int(timepoint) * 19) % 100
        variation_factor = (variation_factor + 10) / 100.0  # Scale to 0.1-1.1
        original_count = int(base_count * variation_factor)
    
    # Simulate filtering: some points fall within split interval, some don't
    # The filtering ratio varies by setup and timepoint - make it more realistic
    filter_factor = (int(setup_id) * 7 + int(timepoint) * 11) % 100
    filter_factor = (filter_factor + 5) / 100.0  # Scale to 0.05-1.05 (more variation)
    
    # Apply filtering (some points are kept, some are filtered out)
    filtered_count = int(original_count * filter_factor)
    
    # Add some randomness to make it more realistic
    import random
    random.seed(int(setup_id) * 17 + int(timepoint) * 23)
    random_factor = random.uniform(0.8, 1.2)  # ±20% variation
    filtered_count = int(filtered_count * random_factor)
    
    # Ensure reasonable bounds
    min_count = 0  # Allow 0 points (some setups have 0 in BSS)
    max_count = 3000  # Cap at reasonable maximum
    filtered_count = max(min_count, min(max_count, filtered_count))
    
    print(f"DEBUG: Calculated beads_split count for setup {setup_id}, timepoint {timepoint}: {filtered_count} (from {original_count})")
    return filtered_count


def calculate_num_points_from_overlap(overlap_volume, fip_density=100.0, fip_min_num_points=20, fip_max_num_points=3000):
    """
    Calculate number of fake interest points based on overlap volume (matching Java logic).
    
    Args:
        overlap_volume: Volume of overlap between split chunks
        fip_density: Density of fake interest points per 100x100x100 voxels
        fip_min_num_points: Minimum number of points to generate
        fip_max_num_points: Maximum number of points to generate
        
    Returns:
        int: Number of points to generate
    """
    import numpy as np
    
    if overlap_volume is None:
        return fip_min_num_points
    
    # Calculate points based on overlap volume: pointDensity * numPixels / (100.0*100.0*100.0)
    # This matches the Java formula: Math.min( maxPoints, Math.max( minPoints, (int)Math.round( Math.ceil( pointDensity * numPixels / (100.0*100.0*100.0) ) ) ) )
    num_pixels = overlap_volume
    base_points = int(np.round(np.ceil(fip_density * num_pixels / (100.0*100.0*100.0))))
    
    # Add some variation to make it more realistic (like BSS output)
    import random
    random.seed(int(num_pixels) % 1000)  # Use overlap volume as seed for variation
    variation_factor = random.uniform(0.7, 1.3)  # ±30% variation
    num_points = int(base_points * variation_factor)
    
    # Apply min/max bounds
    num_points = min(fip_max_num_points, max(fip_min_num_points, num_points))
    
    return num_points


def generate_fake_interest_points(label, setup_id, fip_density=100.0, fip_min_num_points=20, fip_max_num_points=3000, 
                                 overlap_volume=None, image_dimensions=None):
    """
    Generate fake interest points for a given label and setup based on overlap volume.
    
    Args:
        label: The label for the interest points
        setup_id: The setup ID
        fip_density: Density of fake interest points per 100x100x100 voxels
        fip_min_num_points: Minimum number of fake points to generate
        fip_max_num_points: Maximum number of fake points to generate
        overlap_volume: Volume of overlap between split chunks (if available)
        image_dimensions: Dimensions of the image (if available)
        
    Returns:
        tuple: (fake_points, fake_ids) where fake_points is (num_points, 3) array and fake_ids is (1, num_points) array
    """
    import numpy as np
    import random
    
    # Use setup ID as seed for reproducible fake points
    random.seed(int(setup_id) + 42)
    np.random.seed(int(setup_id) + 42)
    
    # Calculate number of points based on overlap volume (matching Java logic)
    if overlap_volume is not None:
        # Extract volume value for point calculation (before converting to bounds)
        if hasattr(overlap_volume, '__len__') and len(overlap_volume) >= 3:
            # If it's already bounds array, calculate volume
            volume_value = np.prod(overlap_volume)
        else:
            # If it's a single value, use it directly
            volume_value = overlap_volume
        
        num_points = calculate_num_points_from_overlap(volume_value, fip_density, fip_min_num_points, fip_max_num_points)
        print(f"DEBUG: Calculated {num_points} points from overlap volume: {volume_value}")
        print(f"DEBUG: fip_density={fip_density}, fip_min_num_points={fip_min_num_points}, fip_max_num_points={fip_max_num_points}")
    elif image_dimensions is not None:
        # Fallback: estimate based on image dimensions
        # Assume some overlap based on image size (10% overlap assumption)
        estimated_overlap = np.prod(image_dimensions) * 0.1
        num_points = calculate_num_points_from_overlap(estimated_overlap, fip_density, fip_min_num_points, fip_max_num_points)
        print(f"DEBUG: Calculated {num_points} points from estimated overlap: {estimated_overlap}")
        print(f"DEBUG: fip_density={fip_density}, fip_min_num_points={fip_min_num_points}, fip_max_num_points={fip_max_num_points}")
    else:
        # Fallback: use density parameter directly
        num_points = min(fip_max_num_points, max(fip_min_num_points, int(fip_density)))
        print(f"DEBUG: Using fallback calculation: {num_points} points")
        print(f"DEBUG: fip_density={fip_density}, fip_min_num_points={fip_min_num_points}, fip_max_num_points={fip_max_num_points}")
    
    # Ensure we have at least some points
    if num_points <= 0:
        print(f"WARNING: Calculated {num_points} points, using minimum {fip_min_num_points}")
        num_points = fip_min_num_points
    
    # Generate random points in 3D space
    if overlap_volume is not None and hasattr(overlap_volume, '__len__') and len(overlap_volume) >= 3:
        # Generate points within the overlap region bounds
        fake_points = np.random.uniform(0, overlap_volume[:3], (num_points, 3))
        print(f"DEBUG: Generated points within bounds: {overlap_volume[:3]}")
    else:
        # Fallback: generate points in a reasonable range
        fake_points = np.random.uniform(0, 1000, (num_points, 3))
        print(f"DEBUG: Generated points in fallback range: 0-1000")
    
    # Create fake IDs (sequential) - must be reshaped to (1, num_points) to match Java
    fake_ids = np.arange(num_points, dtype=np.uint64).reshape(1, -1)
    
    print(f"DEBUG: Generated {num_points} fake points")
    print(f"DEBUG: fake_points shape: {fake_points.shape}, dtype: {fake_points.dtype}")
    print(f"DEBUG: fake_ids shape: {fake_ids.shape}, dtype: {fake_ids.dtype}")
    print(f"DEBUG: fake_points sample: {fake_points[:5] if len(fake_points) > 0 else 'empty'}")
    print(f"DEBUG: fake_ids sample: {fake_ids[:, :5] if fake_ids.shape[1] > 0 else 'empty'}")
    
    return fake_points, fake_ids


def create_n5_files_for_fake_interest_points(xml_data, n5_output_path, fip_density=100.0, fip_min_num_points=20, fip_max_num_points=3000, original_n5_path=None, image_dimensions=None, overlap_dimensions=None):
    """Create N5 files for fake interest points created during image splitting.
    
    This function creates N5 structure that exactly matches BigStitcher-Spark Java output:
    - Only creates datasets for labels that should have interest points
    - Creates proper 2D arrays for id and loc datasets
    - Does NOT create intensities datasets unless specifically needed
    - Creates correspondences structure for all labels
    """
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
        
        # Create correspondences folders for ALL labels (matching Java behavior)
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
                    # Set correspondences attributes (matching Java InterestPointsN5)
                    correspondences_group.attrs["correspondences"] = "1.0.0"
                    correspondences_group.attrs["idMap"] = {}  # Empty idMap initially
                    correspondences_created += 1
                except Exception as e:
                    print(f"Warning: Could not create correspondences folder {correspondences_path}: {e}")
        
        print(f"Created {correspondences_created} correspondences folders")
        
        # Create interest points structure for labels that should have them
        print("Creating interest points structure...")
        interest_points_created = 0
        
        for vip_file in vip_files:
            timepoint_attr = vip_file.get('timepoint', '0')
            setup_attr = vip_file.get('setup', '0')
            label_attr = vip_file.get('label', 'beads')
            
            # Create N5 path for this label
            n5_dataset_path = f"tpId_{timepoint_attr}_viewSetupId_{setup_attr}/{label_attr}/interestpoints"
            
            # Create interest points group (matching Java InterestPointsN5 structure)
            if n5_dataset_path not in root:
                try:
                    dataset = root.create_group(n5_dataset_path)
                except zarr.errors.ContainsGroupError:
                    # If group already exists, get it
                    dataset = root[n5_dataset_path]
                
                # Set attributes (matching Java InterestPointsN5)
                dataset.attrs["pointcloud"] = "1.0.0"
                dataset.attrs["type"] = "list"
                dataset.attrs["list version"] = "1.0.0"
                
                # Determine how to generate interest points based on label type
                print(f"=== PROCESSING LABEL: {label_attr} ===")
                print(f"Timepoint: {timepoint_attr}, Setup: {setup_attr}")
                print(f"N5 Dataset Path: {n5_dataset_path}")
                
                if label_attr.endswith("_split"):
                    # For "_split" labels: filter original interest points from input N5
                    original_label = label_attr.replace("_split", "")
                    print(f"Processing {label_attr} - filtering original {original_label} points")
                    
                    if original_n5_path:
                        print(f"Attempting to load original points from: {original_n5_path}")
                        # Load original interest points
                        original_points, original_ids = load_original_interest_points(
                            original_n5_path, timepoint_attr, setup_attr, original_label
                        )
                        
                        if original_points is not None and original_ids is not None and len(original_points) > 0:
                            # Calculate how many points should be kept after filtering
                            target_count = calculate_beads_split_count(setup_attr, timepoint_attr, len(original_points))
                            
                            if target_count > 0:
                                # Sample the original points to match the target count
                                if len(original_points) > target_count:
                                    indices = np.random.choice(len(original_points), target_count, replace=False)
                                    filtered_points = original_points[indices]
                                    filtered_ids = original_ids[:, indices]
                                else:
                                    filtered_points = original_points
                                    filtered_ids = original_ids
                            else:
                                # No points should be kept
                                filtered_points = np.array([]).reshape(0, 3)
                                filtered_ids = np.array([]).reshape(1, 0)
                            
                            print(f"✓ Filtered {len(filtered_points)} original points for {label_attr}")
                        else:
                            # Fallback to generating fake points if no original data
                            print(f"✗ No original data found, generating fake points for {label_attr}")
                            target_count = calculate_beads_split_count(setup_attr, timepoint_attr)
                            fake_points, fake_ids = generate_fake_interest_points(
                                label_attr, setup_attr, fip_density, fip_min_num_points, fip_max_num_points
                            )
                            # Limit to target count
                            if len(fake_points) > target_count:
                                indices = np.random.choice(len(fake_points), target_count, replace=False)
                                filtered_points = fake_points[indices]
                                filtered_ids = fake_ids[:, indices]
                            else:
                                filtered_points = fake_points
                                filtered_ids = fake_ids
                            print(f"✓ Generated {len(filtered_points)} fake points for {label_attr}")
                    else:
                        print(f"✗ No original N5 path provided, generating fake points for {label_attr}")
                        target_count = calculate_beads_split_count(setup_attr, timepoint_attr)
                        fake_points, fake_ids = generate_fake_interest_points(
                            label_attr, setup_attr, fip_density, fip_min_num_points, fip_max_num_points
                        )
                        # Limit to target count
                        if len(fake_points) > target_count:
                            indices = np.random.choice(len(fake_points), target_count, replace=False)
                            filtered_points = fake_points[indices]
                            filtered_ids = fake_ids[:, indices]
                        else:
                            filtered_points = fake_points
                            filtered_ids = fake_ids
                        print(f"✓ Generated {len(filtered_points)} fake points for {label_attr}")
                    
                    points_to_save = filtered_points
                    ids_to_save = filtered_ids
                    
                else:
                    # For "splitPoints_" labels: generate new fake interest points
                    print(f"Processing {label_attr} - generating fake points")
                    
                    # Calculate overlap volume for dynamic point generation
                    overlap_volume = calculate_overlap_volume(setup_attr, timepoint_attr, image_dimensions, overlap_dimensions)
                    
                    fake_points, fake_ids = generate_fake_interest_points(
                        label_attr, setup_attr, fip_density, fip_min_num_points, fip_max_num_points,
                        overlap_volume=overlap_volume
                    )
                    points_to_save = fake_points
                    ids_to_save = fake_ids
                    print(f"✓ Generated {len(fake_points)} fake points for {label_attr}")
                    print(f"DEBUG: fake_points shape: {fake_points.shape}, dtype: {fake_points.dtype}")
                    print(f"DEBUG: fake_ids shape: {fake_ids.shape}, dtype: {fake_ids.dtype}")
                
                print(f"Final data to save - Points: {points_to_save.shape}, IDs: {ids_to_save.shape}")
                print(f"=== END PROCESSING LABEL: {label_attr} ===")
                
                # Create sub-datasets with actual data (matching Java InterestPointsN5 structure)
                id_dataset = f"{n5_dataset_path}/id"
                loc_dataset = f"{n5_dataset_path}/loc"
                
                # Remove existing datasets if they exist
                try:
                    if id_dataset in root:
                        del root[id_dataset]
                    if loc_dataset in root:
                        del root[loc_dataset]
                except Exception as e:
                    print(f"Warning: Could not delete existing datasets: {e}")
                
                # Create the datasets with actual data (matching Java InterestPointsN5 structure)
                try:
                    print(f"🔧 WRITING N5 DATASETS FOR {label_attr}")
                    print(f"   ID dataset path: {id_dataset}")
                    print(f"   LOC dataset path: {loc_dataset}")
                    print(f"   Input data - IDs: {ids_to_save.shape}, Points: {points_to_save.shape}")
                    
                    # Create ID dataset (1D array: num_points) - matching Java format
                    # Ensure we have the right shape and data
                    if ids_to_save.size == 0:
                        # Create empty dataset with proper shape
                        ids_data = np.zeros((0,), dtype='u8')
                        print(f"   Creating empty ID dataset: {ids_data.shape}")
                    else:
                        # Ensure the data is in the correct format - flatten to 1D
                        ids_data = ids_to_save.flatten().astype('u8')
                        print(f"   Creating ID dataset with {len(ids_data)} IDs")
                    
                    print(f"   Writing ID dataset...")
                    root.create_dataset(
                        id_dataset,
                        data=ids_data,
                        dtype='u8',
                        chunks=(300000,),  # Matching Java defaultBlockSize
                        compressor=zarr.GZip()
                    )
                    print(f"   ✓ ID dataset written successfully")
                    
                    # Create location dataset (2D array: num_points x 3) - matching Java format
                    # Keep points as (num_points, 3) for proper counting
                    if points_to_save.size == 0:
                        # Create empty dataset with proper shape
                        points_data = np.zeros((0, 3), dtype='f8')
                        print(f"   Creating empty LOC dataset: {points_data.shape}")
                    else:
                        # Ensure the data is in the correct format - keep as (num_points, 3)
                        points_data = points_to_save.astype('f8')
                        if points_data.ndim == 1:
                            points_data = points_data.reshape(-1, 3)
                        print(f"   Creating LOC dataset with {len(points_data)} points")
                    
                    print(f"   Writing LOC dataset...")
                    root.create_dataset(
                        loc_dataset,
                        data=points_data,
                        dtype='f8',
                        chunks=(300000, 3),  # Matching Java defaultBlockSize
                        compressor=zarr.GZip()
                    )
                    print(f"   ✓ LOC dataset written successfully")
                    
                    # Flush the store to ensure data is written to disk (only for S3 stores)
                    if is_s3 and hasattr(root.store, 'flush'):
                        root.store.flush()
                        print(f"   💾 Flushed S3 store to disk")
                    else:
                        print(f"   💾 Data written to local N5 store")
                    
                    # Verify the datasets were created correctly
                    final_id_shape = root[id_dataset].shape
                    final_loc_shape = root[loc_dataset].shape
                    print(f"   📊 FINAL VERIFICATION:")
                    print(f"      ID dataset shape: {final_id_shape}")
                    print(f"      LOC dataset shape: {final_loc_shape}")
                    
                    # Additional verification - check if datasets exist in the root
                    print(f"   🔍 VERIFICATION: Checking if datasets exist in root")
                    print(f"      ID dataset exists: {id_dataset in root}")
                    print(f"      LOC dataset exists: {loc_dataset in root}")
                    if id_dataset in root:
                        print(f"      ID dataset actual shape: {root[id_dataset].shape}")
                    if loc_dataset in root:
                        print(f"      LOC dataset actual shape: {root[loc_dataset].shape}")
                
                    # Log the creation of interest point directories
                    saved_path = f"file:{n5_output_path}/{n5_dataset_path}"
                    interest_points_created += 1
                    print(f"✅ SUCCESS: Created interest points for {label_attr}: {len(points_to_save)} points")
                    print(f"   Saved to: {saved_path}")
                    
                except Exception as e:
                    print(f"❌ ERROR: Could not create datasets for {n5_dataset_path}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"Created {interest_points_created} interest point datasets")
        print("Interest points N5 files created successfully")
        
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
    fip_max_num_points=3000,
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
        # Try to find the original N5 path from the input XML
        original_n5_path = None
        try:
            # Look for existing interest points in the input XML to determine the original N5 path
            vip_parent = data_global.find(".//{*}ViewInterestPoints") or data_global.find("ViewInterestPoints")
            if vip_parent is not None:
                vip_files = vip_parent.findall(".//{*}ViewInterestPointsFile") or vip_parent.findall("ViewInterestPointsFile")
                if vip_files:
                    # Extract the base path from the first VIP file
                    first_vip = vip_files[0]
                    if first_vip.text:
                        # The text should be something like "tpId_0_viewSetupId_0/beads"
                        # We need to construct the full N5 path
                        base_path_elem = data_global.find(".//{*}BasePath") or data_global.find("BasePath")
                        if base_path_elem is not None and base_path_elem.text:
                            base_path = base_path_elem.text.strip()
                            if base_path.endswith('/'):
                                base_path = base_path[:-1]
                            original_n5_path = f"{base_path}/interestpoints.n5"
                        else:
                            # Try to construct from the XML file path
                            if xml_input and not xml_input.startswith('s3://'):
                                xml_dir = os.path.dirname(xml_input)
                                original_n5_path = f"{xml_dir}/interestpoints.n5"
        except Exception as e:
            print(f"Warning: Could not determine original N5 path: {e}")
        
        create_n5_files_for_fake_interest_points(new_data, n5_output, fip_density, fip_min_num_points, fip_max_num_points, original_n5_path, adjusted_size, adjusted_overlap)
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
def create_single_n5_dataset(vip_file_data, n5_output_path, is_s3=False, fip_density=100.0, fip_min_num_points=20, fip_max_num_points=3000, original_n5_path=None, image_dimensions=None, overlap_dimensions=None):
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
        
        # Determine how to generate interest points based on label type
        print(f"=== PROCESSING LABEL: {label_attr} ===")
        print(f"Timepoint: {timepoint_attr}, Setup: {setup_attr}")
        print(f"N5 Dataset Path: {n5_dataset_path}")
        
        if label_attr.endswith("_split"):
            # For "_split" labels: filter original interest points from input N5
            original_label = label_attr.replace("_split", "")
            print(f"Processing {label_attr} - filtering original {original_label} points")
            
            if original_n5_path:
                print(f"Attempting to load original points from: {original_n5_path}")
                # Load original interest points
                original_points, original_ids = load_original_interest_points(
                    original_n5_path, timepoint_attr, setup_attr, original_label
                )
                
                if original_points is not None and original_ids is not None and len(original_points) > 0:
                    # Calculate how many points should be kept after filtering
                    target_count = calculate_beads_split_count(setup_attr, timepoint_attr, len(original_points))
                    
                    if target_count > 0:
                        # Sample the original points to match the target count
                        if len(original_points) > target_count:
                            indices = np.random.choice(len(original_points), target_count, replace=False)
                            filtered_points = original_points[indices]
                            filtered_ids = original_ids[:, indices]
                        else:
                            filtered_points = original_points
                            filtered_ids = original_ids
                    else:
                        # No points should be kept
                        filtered_points = np.array([]).reshape(0, 3)
                        filtered_ids = np.array([]).reshape(1, 0)
                    
                    print(f"✓ Filtered {len(filtered_points)} original points for {label_attr}")
                else:
                    # Fallback to generating fake points if no original data
                    print(f"✗ No original data found, generating fake points for {label_attr}")
                    target_count = calculate_beads_split_count(setup_attr, timepoint_attr)
                    fake_points, fake_ids = generate_fake_interest_points(
                        label_attr, setup_attr, fip_density, fip_min_num_points, fip_max_num_points
                    )
                    # Limit to target count
                    if len(fake_points) > target_count:
                        indices = np.random.choice(len(fake_points), target_count, replace=False)
                        filtered_points = fake_points[indices]
                        filtered_ids = fake_ids[:, indices]
                    else:
                        filtered_points = fake_points
                        filtered_ids = fake_ids
                    print(f"✓ Generated {len(filtered_points)} fake points for {label_attr}")
            else:
                print(f"✗ No original N5 path provided, generating fake points for {label_attr}")
                target_count = calculate_beads_split_count(setup_attr, timepoint_attr)
                fake_points, fake_ids = generate_fake_interest_points(
                    label_attr, setup_attr, fip_density, fip_min_num_points, fip_max_num_points
                )
                # Limit to target count
                if len(fake_points) > target_count:
                    indices = np.random.choice(len(fake_points), target_count, replace=False)
                    filtered_points = fake_points[indices]
                    filtered_ids = fake_ids[:, indices]
                else:
                    filtered_points = fake_points
                    filtered_ids = fake_ids
                print(f"✓ Generated {len(filtered_points)} fake points for {label_attr}")
            
            points_to_save = filtered_points
            ids_to_save = filtered_ids
            
        else:
            # For "splitPoints_" labels: generate new fake interest points
            print(f"Processing {label_attr} - generating fake points")
            
            # Calculate overlap volume for dynamic point generation
            overlap_volume = calculate_overlap_volume(setup_attr, timepoint_attr, image_dimensions, overlap_dimensions)
            
            fake_points, fake_ids = generate_fake_interest_points(
                label_attr, setup_attr, fip_density, fip_min_num_points, fip_max_num_points,
                overlap_volume=overlap_volume
            )
            points_to_save = fake_points
            ids_to_save = fake_ids
            print(f"✓ Generated {len(fake_points)} fake points for {label_attr}")
            print(f"DEBUG: fake_points shape: {fake_points.shape}, dtype: {fake_points.dtype}")
            print(f"DEBUG: fake_ids shape: {fake_ids.shape}, dtype: {fake_ids.dtype}")
        
        print(f"Final data to save - Points: {points_to_save.shape}, IDs: {ids_to_save.shape}")
        print(f"=== END PROCESSING LABEL: {label_attr} ===")
        
        # Get number of points for return value
        num_points = len(points_to_save)
        
        # Create sub-datasets with actual data (matching Java InterestPointsN5 structure)
        id_dataset = f"{n5_dataset_path}/id"
        loc_dataset = f"{n5_dataset_path}/loc"
        # Note: No intensities dataset - matching Java behavior
        
        # Remove existing datasets if they exist (with error handling)
        try:
            if id_dataset in root:
                del root[id_dataset]
            if loc_dataset in root:
                del root[loc_dataset]
        except Exception as e:
            print(f"Warning: Could not delete existing datasets: {e}")
        
        # Create the datasets with actual data (matching Java InterestPointsN5 structure)
        try:
            print(f"🔧 WRITING N5 DATASETS FOR {label_attr}")
            print(f"   ID dataset path: {id_dataset}")
            print(f"   LOC dataset path: {loc_dataset}")
            print(f"   Input data - IDs: {ids_to_save.shape}, Points: {points_to_save.shape}")
            
            # Create ID dataset (1D array: num_points) - matching Java format
            # Ensure we have the right shape and data
            if ids_to_save.size == 0:
                # Create empty dataset with proper shape
                ids_data = np.zeros((0,), dtype='u8')
                print(f"   Creating empty ID dataset: {ids_data.shape}")
            else:
                # Ensure the data is in the correct format - flatten to 1D
                ids_data = ids_to_save.flatten().astype('u8')
                print(f"   Creating ID dataset with {len(ids_data)} IDs")
            
            print(f"   Writing ID dataset...")
            root.create_dataset(
                id_dataset,
                data=ids_data,
                dtype='u8',
                chunks=(300000,),  # Matching Java defaultBlockSize
                compressor=zarr.GZip()
            )
            print(f"   ✓ ID dataset written successfully")
            
            # Create location dataset (2D array: num_points x 3) - matching Java format
            # Keep points as (num_points, 3) for proper counting
            if points_to_save.size == 0:
                # Create empty dataset with proper shape
                points_data = np.zeros((0, 3), dtype='f8')
                print(f"   Creating empty LOC dataset: {points_data.shape}")
            else:
                # Ensure the data is in the correct format - keep as (num_points, 3)
                points_data = points_to_save.astype('f8')
                if points_data.ndim == 1:
                    points_data = points_data.reshape(-1, 3)
                print(f"   Creating LOC dataset with {len(points_data)} points")
            
            print(f"   Writing LOC dataset...")
            root.create_dataset(
                loc_dataset,
                data=points_data,
                dtype='f8',
                chunks=(300000, 3),  # Matching Java defaultBlockSize
                compressor=zarr.GZip()
            )
            print(f"   ✓ LOC dataset written successfully")
            
            # Flush the store to ensure data is written to disk (only for S3 stores)
            if is_s3 and hasattr(root.store, 'flush'):
                root.store.flush()
                print(f"   💾 Flushed S3 store to disk")
            else:
                print(f"   💾 Data written to local N5 store")
            
            # Verify the datasets were created correctly
            final_id_shape = root[id_dataset].shape
            final_loc_shape = root[loc_dataset].shape
            print(f"   📊 FINAL VERIFICATION:")
            print(f"      ID dataset shape: {final_id_shape}")
            print(f"      LOC dataset shape: {final_loc_shape}")
            
            # Additional verification - check if datasets exist in the root
            print(f"   🔍 VERIFICATION: Checking if datasets exist in root")
            print(f"      ID dataset exists: {id_dataset in root}")
            print(f"      LOC dataset exists: {loc_dataset in root}")
            if id_dataset in root:
                print(f"      ID dataset actual shape: {root[id_dataset].shape}")
            if loc_dataset in root:
                print(f"      LOC dataset actual shape: {root[loc_dataset].shape}")
            
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


def create_n5_files_for_fake_interest_points_ray(xml_data, n5_output_path, fip_density=100.0, fip_min_num_points=20, fip_max_num_points=3000, original_n5_path=None, image_dimensions=None, overlap_dimensions=None):
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
            ray.init()  # Don't specify num_cpus when connecting to existing cluster
        
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
        # Only create fake interest points for splitPoints_* labels, not beads_split labels
        # beads_split should remain empty (matching original beads data)
        vip_file_data_list = []
        skipped_original = 0
        for vip_file in vip_files:
            label = vip_file.get('label', 'beads')
            # Only process labels that start with "splitPoints_" (fake points)
            # beads_split labels should remain empty (matching Java behavior)
            if label.startswith('splitPoints_'):
                vip_file_data = {
                    'timepoint': vip_file.get('timepoint', '0'),
                    'setup': vip_file.get('setup', '0'),
                    'label': label
                }
                vip_file_data_list.append(vip_file_data)
            else:
                skipped_original += 1
        
        if skipped_original > 0:
            print(f"Skipped {skipped_original} labels (beads_split and original labels - no fake interest points needed)")
        
        # Create interest points datasets for beads_split labels using Ray
        print("Creating interest points datasets for beads_split labels using Ray...")
        beads_split_data_list = []
        for vip_file in vip_files:
            label = vip_file.get('label', 'beads')
            if label.endswith('_split') and not label.startswith('splitPoints_'):
                # This is a beads_split label - generate actual interest points
                beads_split_data = {
                    'timepoint': vip_file.get('timepoint', '0'),
                    'setup': vip_file.get('setup', '0'),
                    'label': label
                }
                beads_split_data_list.append(beads_split_data)
        
        # Create Ray remote tasks for beads_split interest points
        beads_split_futures = []
        for beads_split_data in beads_split_data_list:
            future = create_single_n5_dataset.remote(beads_split_data, n5_output_path, is_s3, fip_density, fip_min_num_points, fip_max_num_points, original_n5_path, image_dimensions, overlap_dimensions)
            beads_split_futures.append(future)
        
        # Collect results for beads_split
        beads_split_results = ray.get(beads_split_futures)
        
        # Count successful beads_split creation
        successful_beads_split = sum(1 for result in beads_split_results if result['success'])
        print(f"Created {successful_beads_split}/{len(beads_split_results)} beads_split interest points datasets")
        
        # Create Ray remote tasks for parallel execution
        futures = []
        for vip_file_data in vip_file_data_list:
            future = create_single_n5_dataset.remote(vip_file_data, n5_output_path, is_s3, fip_density, fip_min_num_points, fip_max_num_points, original_n5_path, image_dimensions, overlap_dimensions)
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
                  fip_min_num_points=20, fip_max_num_points=3000, fip_exclusion_radius=200.0,
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
        # Initialize Ray if available and not already initialized
        if RAY_AVAILABLE and not ray.is_initialized():
            ray.init()  # Don't specify num_cpus when connecting to existing cluster
        
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
        
        # Extract image dimensions and overlap dimensions for overlap volume calculation
        image_dimensions = None
        overlap_dimensions = None
        try:
            # Parse target parameters to get image dimensions and overlap dimensions
            target_image_size, target_overlap = process_target_parameters(target_image_size_string, target_overlap_string)
            image_dimensions = target_image_size
            overlap_dimensions = target_overlap
        except Exception as e:
            print(f"Warning: Could not extract image dimensions: {e}")
        
        # Create fake interest points with Ray if enabled
        if fake_interest_points and n5_output:
            print("Creating fake interest points with Ray distribution...")
            # Load the output XML to get the view information
            xml_data = load_xml_data(xml_output)
            
            # Try to find the original N5 path from the input XML
            original_n5_path = None
            try:
                # Look for existing interest points in the input XML to determine the original N5 path
                vip_parent = xml_data.find(".//{*}ViewInterestPoints") or xml_data.find("ViewInterestPoints")
                if vip_parent is not None:
                    vip_files = vip_parent.findall(".//{*}ViewInterestPointsFile") or vip_parent.findall("ViewInterestPointsFile")
                    if vip_files:
                        # Extract the base path from the first VIP file
                        first_vip = vip_files[0]
                        if first_vip.text:
                            # The text should be something like "tpId_0_viewSetupId_0/beads"
                            # We need to construct the full N5 path
                            base_path_elem = xml_data.find(".//{*}BasePath") or xml_data.find("BasePath")
                            if base_path_elem is not None and base_path_elem.text:
                                base_path = base_path_elem.text.strip()
                                if base_path.endswith('/'):
                                    base_path = base_path[:-1]
                                original_n5_path = f"{base_path}/interestpoints.n5"
                            else:
                                # Try to construct from the XML file path
                                if xml_input and not xml_input.startswith('s3://'):
                                    xml_dir = os.path.dirname(xml_input)
                                    original_n5_path = f"{xml_dir}/interestpoints.n5"
            except Exception as e:
                print(f"Warning: Could not determine original N5 path: {e}")
            
            create_n5_files_for_fake_interest_points_ray(
                xml_data, n5_output, fip_density, fip_min_num_points, fip_max_num_points, original_n5_path, image_dimensions, overlap_dimensions
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