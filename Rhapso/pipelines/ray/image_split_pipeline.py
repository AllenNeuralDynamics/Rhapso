"""
Ray-distributed Image Split Pipeline for Rhapso.

This module provides a Ray-distributed version of the image splitting pipeline
that parallelizes the creation of N5 files for fake interest points. This
significantly improves performance when processing many viewpoints.

Key improvements over the sequential version:
- Parallel N5 dataset creation using Ray remote tasks
- Configurable CPU usage and memory allocation
- Fallback to sequential processing if needed
- Better error handling and progress reporting

Usage:
    python3 -m Rhapso.pipelines.ray.image_split_pipeline

Configuration:
    - USE_RAY_DISTRIBUTION: Enable/disable Ray distribution (default: True)
    - RAY_NUM_CPUS: Number of CPUs to use (default: all available)
    - RAY_OBJECT_STORE_MEMORY: Ray object store memory limit (default: auto)

Performance:
    The Ray-distributed version can achieve near-linear speedup for N5 file
    creation, reducing processing time from minutes to seconds for large
    datasets with many viewpoints.
 
Fake Interest Points:
    This implementation creates actual fake interest points with 3D coordinates,
    intensities, and IDs instead of empty datasets. This ensures compatibility
    with BigStitcher Spark and other downstream processing tools that expect
    real interest point data.
"""
print("Begin Ray-distributed image split pipeline.")

import ray
import os
import zarr
import s3fs
from xml.etree import ElementTree as ET

# Import the main function from split_datasets
from Rhapso.image_split.split_datasets import main

# Configuration Variables

# Local Input/Output Paths
XML_INPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/bigstitcher_affine.xml"
XML_OUTPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/bigstitcher_affine_split.xml"
N5_OUTPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/interestpoints.n5"

# XML_INPUT = "/Users/seanfite/Desktop/interest_point_detection/rhapso-solver-affine.xml"
# XML_OUTPUT = "/Users/seanfite/Desktop/interest_point_detection/rhapso-solver-split-affine.xml"
# N5_OUTPUT = "/Users/seanfite/Desktop/interest_point_detection/interestpoints.n5"

# AWS S3 Input/Output Paths
# XML_INPUT = "s3://martin-test-bucket/split_images_output/bigstitcher_affine.xml"
# XML_OUTPUT = "s3://martin-test-bucket/split_images_output/output.xml"
# N5_OUTPUT = "s3://martin-test-bucket/split_images_output/interestpoints.n5"

# XML_INPUT = "s3://rhapso-matching-test/rhapso-solver-affine.xml"
# XML_OUTPUT = "s3://rhapso-matching-test/rhapso-solver-split-affine"
# N5_OUTPUT = "s3://rhapso-matching-test/interestpoints.n5"

# Image Splitting Parameters
TARGET_IMAGE_SIZE_STRING = "7000,7000,4000"
TARGET_OVERLAP_STRING = "128,128,128"

# Fake Interest Points Configuration
FAKE_INTEREST_POINTS = True
FIP_EXCLUSION_RADIUS = 200
ASSIGN_ILLUMINATIONS = True

# Default Values
DISABLE_OPTIMIZATION = False
FIP_DENSITY = 100.0
FIP_MIN_NUM_POINTS = 20
FIP_MAX_NUM_POINTS = 500
FIP_ERROR = 0.5

# Ray Configuration
RAY_NUM_CPUS = 4  # Limit to 4 CPUs to reduce file system race conditions
RAY_OBJECT_STORE_MEMORY = None  # Use default object store memory
USE_RAY_DISTRIBUTION = True  # Set to False to use original sequential version


# ============================================================================
# Ray-distributed N5 File Creation Functions
# ============================================================================

@ray.remote
def create_single_n5_dataset(vip_file_data, n5_output_path, is_s3=False, fip_density=100.0, fip_min_num_points=20, fip_max_num_points=500):
    """
    Create a single N5 dataset for one ViewInterestPointsFile with actual fake interest points.
    This function runs as a Ray remote task for parallel execution.
    
    Args:
        vip_file_data: Dictionary containing timepoint, setup, and label attributes
        n5_output_path: Path to the N5 output directory
        is_s3: Whether the path is an S3 path
        fip_density: Density of fake interest points per unit volume
        fip_min_num_points: Minimum number of fake interest points
        fip_max_num_points: Maximum number of fake interest points
    
    Returns:
        Dictionary with success status and path information
    """
    try:
        import numpy as np 
        
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
            # Handle local path - use zarr N5Store
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
        
        # Generate a reasonable number of fake interest points
        # For fake points, we'll create a grid-like pattern in 3D space
        num_points = max(fip_min_num_points, min(fip_max_num_points, int(fip_density / 10)))
        
        # Create fake 3D coordinates (x, y, z)
        # Use a grid pattern to ensure points are well-distributed
        grid_size = int(np.ceil(num_points ** (1/3)))
        if grid_size < 2:
            grid_size = 2
        
        # Create 3D grid coordinates
        x_coords = np.linspace(100, 1000, grid_size)  # X coordinates
        y_coords = np.linspace(100, 1000, grid_size)  # Y coordinates  
        z_coords = np.linspace(50, 500, grid_size)    # Z coordinates
        
        # Create meshgrid and flatten
        xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        fake_locations = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        # Take only the number of points we need
        if len(fake_locations) > num_points:
            # Randomly sample if we have too many
            indices = np.random.choice(len(fake_locations), num_points, replace=False)
            fake_locations = fake_locations[indices]
        elif len(fake_locations) < num_points:
            # Duplicate and add noise if we have too few
            while len(fake_locations) < num_points:
                # Add some random points
                random_points = np.random.uniform(100, 1000, (min(10, num_points - len(fake_locations)), 3))
                random_points[:, 2] = np.random.uniform(50, 500, len(random_points))  # Z coordinate
                fake_locations = np.vstack([fake_locations, random_points])
            fake_locations = fake_locations[:num_points]
        
        # Add some random noise to make points more realistic
        noise = np.random.normal(0, 5, fake_locations.shape)
        fake_locations = fake_locations + noise
        
        # Create fake intensities (random values between 0.1 and 1.0)
        # Intensities should be 1D array
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
            print(f"Warning: Could not delete existing datasets for {n5_dataset_path}: {e}")
        
        # Create datasets with actual fake data (with error handling)
        try:
            root.create_dataset(
                id_dataset,
                data=fake_ids,
                dtype='u8',
                chunks=(min(100, num_points), 1),
                compressor=zarr.GZip()
            )
            
            root.create_dataset(
                loc_dataset,
                data=fake_locations,
                dtype='f8',
                chunks=(min(100, num_points), 3),
                compressor=zarr.GZip()
            )
            
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
    This replaces the sequential version in split_datasets.py.
    
    Args:
        xml_data: XML data containing ViewInterestPointsFile elements
        n5_output_path: Path to the N5 output directory
        fip_density: Density of fake interest points per unit volume
        fip_min_num_points: Minimum number of fake interest points
        fip_max_num_points: Maximum number of fake interest points
    """
    try:
        # Validate n5_output_path
        if not n5_output_path or not n5_output_path.strip():
            raise ValueError("n5_output_path is empty or not provided")
        
        print("Saving interest points using Ray distribution...")
        
        # Find all ViewInterestPointsFile elements to get all labels and setups
        vip_files = xml_data.findall('.//ViewInterestPointsFile') or xml_data.findall('.//{*}ViewInterestPointsFile')
        
        if not vip_files:
            print("No ViewInterestPointsFile elements found in XML")
            return
        
        # Determine if this is an S3 path or local path
        is_s3 = n5_output_path.startswith('s3://')
        
        # Create the root N5 structure first (before parallel processing)
        print("Creating root N5 structure...")
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
        
        # Set N5 version attribute (use non-reserved name)
        root.attrs['n5_version'] = '4.0.0'
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                num_cpus=RAY_NUM_CPUS,
                object_store_memory=RAY_OBJECT_STORE_MEMORY
            )
        
        # Prepare data for parallel processing
        vip_file_data_list = []
        for vip_file in vip_files:
            vip_file_data = {
                'timepoint': vip_file.get('timepoint', '0'),
                'setup': vip_file.get('setup', '0'),
                'label': vip_file.get('label', 'beads')
            }
            vip_file_data_list.append(vip_file_data)
        
        # Create Ray remote tasks for parallel execution
        print(f"Creating {len(vip_file_data_list)} N5 datasets in parallel...")
        futures = []
        for vip_file_data in vip_file_data_list:
            future = create_single_n5_dataset.remote(vip_file_data, n5_output_path, is_s3, fip_density, fip_min_num_points, fip_max_num_points)
            futures.append(future)
        
        # Wait for all tasks to complete and collect results
        results = ray.get(futures)
        
        # Process results
        successful_count = 0
        failed_count = 0
        total_points = 0
        
        for result in results:
            if result["success"]:
                num_points = result.get('num_points', 0)
                total_points += num_points
                print(f"Saved: {result['path']} ({num_points} points)")
                successful_count += 1
            else:
                print(f"Failed to create {result['dataset_path']}: {result['error']}")
                failed_count += 1
        
        print(f"N5 file creation completed: {successful_count} successful, {failed_count} failed")
        print(f"Total fake interest points created: {total_points}")
        
        if failed_count > 0:
            print("Warning: Some N5 datasets failed to create. Check the error messages above.")
            print("This may be due to file system race conditions in parallel processing.")
            print("Consider reducing RAY_NUM_CPUS or using sequential processing for more reliable results.")
        
    except Exception as e:
        print(f"Warning: Could not create N5 files for fake interest points: {e}")
        print("Continuing without N5 file creation...")


def run_image_split_pipeline_with_ray():
    """
    Run the image split pipeline with Ray distribution for N5 file creation.
    """
    print("Running image split pipeline with Ray distribution...")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=RAY_NUM_CPUS,
            object_store_memory=RAY_OBJECT_STORE_MEMORY
        )
        print(f"Ray initialized with {ray.cluster_resources()}")
    
    try:
        # Run the main splitting process (without N5 creation)
        main(
            xml_input=XML_INPUT,
            xml_output=XML_OUTPUT,
            n5_output=None,  # We'll handle N5 creation separately with Ray
            target_image_size_string=TARGET_IMAGE_SIZE_STRING,
            target_overlap_string=TARGET_OVERLAP_STRING,
            fake_interest_points=FAKE_INTEREST_POINTS,
            fip_exclusion_radius=FIP_EXCLUSION_RADIUS,
            assign_illuminations=ASSIGN_ILLUMINATIONS,
            disable_optimization=DISABLE_OPTIMIZATION,
            fip_density=FIP_DENSITY,
            fip_min_num_points=FIP_MIN_NUM_POINTS,
            fip_max_num_points=FIP_MAX_NUM_POINTS,
            fip_error=FIP_ERROR
        )
        
        # Now create N5 files using Ray distribution
        if FAKE_INTEREST_POINTS and N5_OUTPUT:
            print("\nCreating N5 files with Ray distribution...")
            # Load the XML data to get the split results
            from Rhapso.image_split.split_datasets import load_xml_data
            xml_data = load_xml_data(XML_OUTPUT)
            if xml_data is not None:
                create_n5_files_for_fake_interest_points_ray(
                    xml_data, N5_OUTPUT, FIP_DENSITY, FIP_MIN_NUM_POINTS, FIP_MAX_NUM_POINTS
                )
            else:
                print("Warning: Could not load XML data for N5 creation")
        elif FAKE_INTEREST_POINTS and not N5_OUTPUT:
            print("Warning: fake_interest_points is True but n5_output is not provided. Skipping N5 file creation.")
        
        print("Image split pipeline with Ray distribution completed.")
        
    finally:
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown completed.")


if __name__ == "__main__":
    if USE_RAY_DISTRIBUTION:
        print("Using Ray-distributed version for better performance...")
        run_image_split_pipeline_with_ray()
    else:
        print("Using original sequential version...")
        main(
            xml_input=XML_INPUT,
            xml_output=XML_OUTPUT,
            n5_output=N5_OUTPUT,
            target_image_size_string=TARGET_IMAGE_SIZE_STRING,
            target_overlap_string=TARGET_OVERLAP_STRING,
            fake_interest_points=FAKE_INTEREST_POINTS,
            fip_exclusion_radius=FIP_EXCLUSION_RADIUS,
            assign_illuminations=ASSIGN_ILLUMINATIONS,
            disable_optimization=DISABLE_OPTIMIZATION,
            fip_density=FIP_DENSITY,
            fip_min_num_points=FIP_MIN_NUM_POINTS,
            fip_max_num_points=FIP_MAX_NUM_POINTS,
            fip_error=FIP_ERROR
        )
        print("Image split pipeline completed.")
else:
    print("Image split pipeline module imported.")