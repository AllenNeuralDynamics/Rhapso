"""
Ray-based parallel image splitting for Rhapso.

This module provides Ray-based parallel processing for image splitting operations,
replicating the pattern used in detection and matching pipelines.
"""

import ray
import numpy as np
from xml.etree import ElementTree as ET
import zarr
import os
import s3fs
import boto3
from io import BytesIO
import time

# Note: Imports are done inside methods to avoid import errors on Ray cluster


class SplitImages:
    def __init__(self, xml_input, xml_output, n5_output, target_image_size_string, 
                 target_overlap_string, fake_interest_points, fip_exclusion_radius,
                 assign_illuminations, disable_optimization, fip_density,
                 fip_min_num_points, fip_max_num_points, fip_error):
        self.xml_input = xml_input
        self.xml_output = xml_output
        self.n5_output = n5_output
        self.target_image_size_string = target_image_size_string
        self.target_overlap_string = target_overlap_string
        self.fake_interest_points = fake_interest_points
        self.fip_exclusion_radius = fip_exclusion_radius
        self.assign_illuminations = assign_illuminations
        self.disable_optimization = disable_optimization
        self.fip_density = fip_density
        self.fip_min_num_points = fip_min_num_points
        self.fip_max_num_points = fip_max_num_points
        self.fip_error = fip_error

    def create_n5_files_parallel(self, xml_data, n5_output_path):
        """Create N5 files for fake interest points using Ray parallel processing."""
        try:
            # Validate n5_output_path
            if not n5_output_path or not n5_output_path.strip():
                raise ValueError("n5_output_path is empty or not provided")
            
            print("Saving interest points with Ray parallel processing...")
            print(f"📊 Ray Cluster Info:")
            print(f"   ├── Cluster Address: {ray.get_runtime_context().gcs_address}")
            print(f"   ├── Dashboard: http://{ray.get_runtime_context().gcs_address.split(':')[0]}:8265")
            print(f"   └── Available Resources: {ray.cluster_resources()}")
            
            # Find all ViewInterestPointsFile elements to get all labels and setups
            vip_files = xml_data.findall('.//ViewInterestPointsFile') or xml_data.findall('.//{*}ViewInterestPointsFile')
            
            if not vip_files:
                print("No ViewInterestPointsFile elements found in XML")
                return
            
            print(f"📋 Found {len(vip_files)} ViewInterestPointsFile elements to process")
            
            # Determine if this is an S3 path or local path
            is_s3 = n5_output_path.startswith('s3://')
            
            # Generate a single shared timestamp for all splitPoints folders
            import time
            shared_timestamp = int(time.time() * 1000)  # milliseconds timestamp
            print(f"Using shared timestamp for splitPoints: {shared_timestamp}")
            
            # Create Ray remote function for parallel N5 file creation with resource requirements
            @ray.remote(num_cpus=1, memory=2*1024*1024*1024)  # 1 CPU, 2GB memory per task
            def create_single_n5_file(vip_file_data, n5_output_path, is_s3, fip_density, fip_min_num_points, fip_max_num_points, shared_timestamp):
                """Create N5 files for both beads_split and splitPoints folders."""
                max_retries = 3
                retry_delay = 1  # seconds
                
                for attempt in range(max_retries):
                    try:
                        timepoint_attr = vip_file_data['timepoint']
                        setup_attr = vip_file_data['setup']
                        label_attr = vip_file_data['label']
                        
                        # Validate attributes
                        if not timepoint_attr or not setup_attr or not label_attr:
                            raise ValueError(f"Missing required attributes: timepoint='{timepoint_attr}', setup='{setup_attr}', label='{label_attr}'")
                        
                        # Create paths - only create splitPoints folder, not beads_split
                        split_points_path = f"tpId_{timepoint_attr}_viewSetupId_{setup_attr}/splitPoints_{shared_timestamp}/interestpoints"
                        
                        # Validate paths are not empty
                        if not split_points_path:
                            raise ValueError(f"Generated empty path: split_points='{split_points_path}'")
                        
                        print(f"[Ray Worker] Attempt {attempt + 1}/{max_retries} - Processing setup {setup_attr}, path: '{split_points_path}'")
                        
                        if is_s3:
                            # Handle S3 path - use zarr with s3fs with optimized settings
                            s3_fs = s3fs.S3FileSystem(
                                default_cache_type='bytes',  # Use bytes cache for better performance
                                default_fill_cache=False,    # Don't fill cache to save memory
                                default_block_size=5*1024*1024  # 5MB blocks for better throughput
                            )
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
                        
                        # Generate fake interest points data
                        num_points = min(fip_max_num_points, max(fip_min_num_points, int(fip_density)))
                        
                        # Generate random interest points with better performance
                        np.random.seed(42)  # For reproducible results
                        interest_points = np.random.rand(num_points, 3).astype(np.float32) * 1000  # Use float32 for better performance
                        point_ids = np.arange(num_points, dtype=np.uint32)  # Use uint32 instead of uint64 for better performance
                        intensities = np.random.rand(num_points).astype(np.float32) * 1000  # Random intensities
                    
                        # Helper function to create interest points structure
                        def create_interest_points_structure(dataset_path):
                            """Create the interest points structure for a given path."""
                            print(f"[Ray Worker] Creating interest points structure for path: '{dataset_path}'")
                            
                            if not dataset_path or dataset_path.strip() == "":
                                raise ValueError(f"Empty dataset_path provided: '{dataset_path}'")
                            
                            if dataset_path not in root:
                                try:
                                    dataset = root.create_group(dataset_path)
                                    print(f"[Ray Worker] Created new group at path: '{dataset_path}'")
                                except zarr.errors.ContainsGroupError:
                                    # If group already exists, get it
                                    dataset = root[dataset_path]
                                    print(f"[Ray Worker] Using existing group at path: '{dataset_path}'")
                                except Exception as e:
                                    raise ValueError(f"Failed to create/access group at path '{dataset_path}': {e}")
                                
                                # Set attributes
                                dataset.attrs["pointcloud"] = "1.0.0"
                                dataset.attrs["type"] = "list"
                                dataset.attrs["list version"] = "1.0.0"
                                
                                # Create sub-datasets
                                id_dataset = f"{dataset_path}/id"
                                loc_dataset = f"{dataset_path}/loc"
                                intensities_dataset = f"{dataset_path}/intensities"
                                
                                print(f"[Ray Worker] Creating datasets: id='{id_dataset}', loc='{loc_dataset}', intensities='{intensities_dataset}'")
                                
                                # Create datasets with optimized chunking and compression
                                chunk_size = min(1000, num_points)  # Larger chunks for better performance
                                
                                if id_dataset not in root:
                                    id_ds = root.create_dataset(
                                        id_dataset,
                                        data=point_ids,
                                        dtype='u4',  # Changed from u8 to u4 for better performance
                                        chunks=(chunk_size,),  
                                        compressor=zarr.Blosc(cname='lz4', clevel=1, shuffle=1)  # Faster compression
                                    )
                                    # Set N5-specific attributes
                                    id_ds.attrs.update({
                                        "dimensions": [num_points],
                                        "blockSize": [chunk_size]
                                    })
                                    print(f"[Ray Worker] Created id dataset at: '{id_dataset}' with {num_points} points")
                                
                                if loc_dataset not in root:
                                    loc_ds = root.create_dataset(
                                        loc_dataset,
                                        data=interest_points,
                                        dtype='f4',  # Changed from f8 to f4 for better performance
                                        chunks=(chunk_size, 3), 
                                        compressor=zarr.Blosc(cname='lz4', clevel=1, shuffle=1)  # Faster compression
                                    )
                                    # Set N5-specific attributes
                                    loc_ds.attrs.update({
                                        "dimensions": [num_points, 3],
                                        "blockSize": [chunk_size, 3]
                                    })
                                    print(f"[Ray Worker] Created loc dataset at: '{loc_dataset}' with {num_points} points")
                                
                                if intensities_dataset not in root:
                                    intensities_ds = root.create_dataset(
                                        intensities_dataset,
                                        data=intensities,
                                        dtype='f4', 
                                        chunks=(chunk_size,),  
                                        compressor=zarr.Blosc(cname='lz4', clevel=1, shuffle=1)  # Faster compression
                                    )
                                    # Set N5-specific attributes
                                    intensities_ds.attrs.update({
                                        "dimensions": [num_points],
                                        "blockSize": [chunk_size]
                                    })
                                    print(f"[Ray Worker] Created intensities dataset at: '{intensities_dataset}' with {num_points} points")
                            else:
                                print(f"[Ray Worker] Path '{dataset_path}' already exists in root, skipping creation")
                    
                        # Helper function to create correspondences structure
                        def create_correspondences_structure(base_path):
                            """Create the correspondences structure for a given path."""
                            correspondences_path = base_path.replace('/interestpoints', '/correspondences')
                            print(f"[Ray Worker] Creating correspondences structure for path: '{correspondences_path}'")
                            
                            if not correspondences_path or correspondences_path.strip() == "":
                                raise ValueError(f"Empty correspondences_path generated from base_path: '{base_path}'")
                            
                            if correspondences_path not in root:
                                try:
                                    dataset = root.create_group(correspondences_path)
                                    print(f"[Ray Worker] Created new correspondences group at path: '{correspondences_path}'")
                                except zarr.errors.ContainsGroupError:
                                    dataset = root[correspondences_path]
                                    print(f"[Ray Worker] Using existing correspondences group at path: '{correspondences_path}'")
                                except Exception as e:
                                    raise ValueError(f"Failed to create/access correspondences group at path '{correspondences_path}': {e}")
                                
                                # Set attributes for correspondences
                                dataset.attrs["pointcloud"] = "1.0.0"
                                dataset.attrs["type"] = "list"
                                dataset.attrs["list version"] = "1.0.0"
                            else:
                                print(f"[Ray Worker] Correspondences path '{correspondences_path}' already exists, skipping creation")
                        
                        # Create splitPoints structure only
                        print(f"[Ray Worker] Creating splitPoints structure for setup {setup_attr}")
                        create_interest_points_structure(split_points_path)
                        create_correspondences_structure(split_points_path)
                        
                        # Return success message
                        saved_paths = f"file:{n5_output_path}/{split_points_path}"
                        print(f"[Ray Worker] Successfully completed setup {setup_attr} on attempt {attempt + 1}")
                        return {'success': True, 'path': saved_paths, 'setup': setup_attr, 'attempt': attempt + 1}
                        
                    except Exception as e:
                        error_msg = f"Attempt {attempt + 1}/{max_retries} failed for setup {setup_attr}: {str(e)}"
                        print(f"[Ray Worker] {error_msg}")
                        
                        if attempt < max_retries - 1:
                            print(f"[Ray Worker] Retrying setup {setup_attr} in {retry_delay} seconds...")
                            import time
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            print(f"[Ray Worker] All {max_retries} attempts failed for setup {setup_attr}")
                            return {'success': False, 'error': str(e), 'setup': setup_attr, 'attempts': max_retries}
            
            # Prepare data for parallel processing
            vip_file_data_list = []
            for vip_file in vip_files:
                vip_file_data_list.append({
                    'timepoint': vip_file.get('timepoint', '0'),
                    'setup': vip_file.get('setup', '0'),
                    'label': vip_file.get('label', 'beads')
                })
            
            # Submit tasks to Ray for parallel processing with better resource management
            print(f"Submitting {len(vip_file_data_list)} Ray tasks for parallel N5 file creation...")
            
            # Submit all tasks at once for better parallelization
            futures = [
                create_single_n5_file.remote(vip_data, n5_output_path, is_s3, self.fip_density, self.fip_min_num_points, self.fip_max_num_points, shared_timestamp)
                for vip_data in vip_file_data_list
            ]
            
            # Collect results with progress tracking and timeout
            print("Waiting for Ray tasks to complete...")
            try:
                results = ray.get(futures, timeout=300)  # 5 minute timeout
            except ray.exceptions.GetTimeoutError:
                print("⚠️  Ray tasks timed out after 5 minutes. Some tasks may still be running.")
                # Get partial results
                results = []
                for i, future in enumerate(futures):
                    try:
                        result = ray.get(future, timeout=1)
                        results.append(result)
                    except:
                        results.append({'success': False, 'error': 'Task timed out', 'setup': f'setup_{i}'})
            
            # Process results and print detailed status
            successful_count = 0
            failed_setups = []
            retry_stats = {}
            
            for result in results:
                if result['success']:
                    print(f"✅ Saved: {result['path']}")
                    successful_count += 1
                    if 'attempt' in result and result['attempt'] > 1:
                        setup_id = result['setup']
                        if setup_id not in retry_stats:
                            retry_stats[setup_id] = 0
                        retry_stats[setup_id] = result['attempt']
                else:
                    print(f"❌ Failed to save setup {result['setup']}: {result['error']}")
                    failed_setups.append({
                        'setup': result['setup'],
                        'error': result['error'],
                        'attempts': result.get('attempts', 1)
                    })
            
            # Print retry statistics
            if retry_stats:
                print(f"\n📊 Retry Statistics:")
                for setup_id, attempts in retry_stats.items():
                    print(f"  Setup {setup_id}: succeeded after {attempts} attempts")
            
            # Print failure summary
            if failed_setups:
                print(f"\n❌ Failed Setups Summary:")
                for failure in failed_setups:
                    print(f"  Setup {failure['setup']}: {failure['error']} (after {failure['attempts']} attempts)")
            
            print(f"\n📈 Final Results: {successful_count}/{len(vip_file_data_list)} files created successfully")
            
            if successful_count < len(vip_file_data_list):
                print(f"⚠️  {len(vip_file_data_list) - successful_count} setups failed to save")
            
        except Exception as e:
            print(f"Warning: Could not create N5 files for fake interest points: {e}")
            print("Continuing without N5 file creation...")

    def run(self):
        """Main entry point for Ray-based image splitting process."""
        print("Beginning Ray-based image splitting...")
        
        # Ensure Ray is initialized and check cluster status
        if not ray.is_initialized():
            print("⚠️  Ray not initialized. Initializing Ray...")
            ray.init(ignore_reinit_error=True)
        
        # Print Ray cluster information
        try:
            print(f"📊 Ray Cluster Status:")
            print(f"   ├── Cluster Address: {ray.get_runtime_context().gcs_address}")
            print(f"   ├── Dashboard: http://{ray.get_runtime_context().gcs_address.split(':')[0]}:8265")
            print(f"   ├── Available Resources: {ray.cluster_resources()}")
            print(f"   └── Node Count: {len(ray.nodes())}")
        except Exception as e:
            print(f"⚠️  Could not get Ray cluster info: {e}")
        
        # Import the existing split_images_main function
        try:
            from Rhapso.image_split.split_datasets import main as split_images_main
        except ImportError as e:
            print(f"Error: Could not import split_images_main: {e}")
            print("Falling back to direct implementation...")
            # Fallback to the original approach if import fails
            self._run_fallback()
            return
        
        # Use the existing split_images_main function for the main processing
        # but intercept the N5 file creation to use Ray parallel processing
        print("Running image splitting with existing split_images_main...")
        
        # Call the original function to create the basic structure and preserve original beads folder
        try:
            split_images_main(
                xml_input=self.xml_input,
                xml_output=self.xml_output,
                n5_output=self.n5_output,
                target_image_size_string=self.target_image_size_string,
                target_overlap_string=self.target_overlap_string,
                fake_interest_points=True,  # Enable fake interest points to create original beads structure
                fip_exclusion_radius=self.fip_exclusion_radius,
                assign_illuminations=self.assign_illuminations,
                disable_optimization=self.disable_optimization,
                fip_density=self.fip_density,
                fip_min_num_points=self.fip_min_num_points,
                fip_max_num_points=self.fip_max_num_points,
                fip_error=self.fip_error
            )
        except Exception as e:
            print(f"Error in split_images_main: {e}")
            raise
        
        # Note: split_images_main already creates the necessary structures including splitPoints
        # No additional Ray-based N5 creation needed to avoid duplication
        print("split_images_main has completed all necessary N5 file creation.")
        print("✅ All N5 structures created successfully by split_images_main")

        print("Ray-based split-images run finished")
        
        # Print output file summary
        print("\n" + "="*60)
        print("📁 OUTPUT FILES SUMMARY")
        print("="*60)
        print(f"📄 XML Output: {self.xml_output}")
        if self.fake_interest_points and self.n5_output:
            print(f"🗃️  N5 Output: {self.n5_output}")
            print(f"   ├── Beads: {self.n5_output}/tpId_*_viewSetupId_*/beads/interestpoints")
            print(f"   ├── Beads Split: {self.n5_output}/tpId_*_viewSetupId_*/beads_split/interestpoints")
            print(f"   └── Split Points: {self.n5_output}/tpId_*_viewSetupId_*/splitPoints_*/interestpoints")
        else:
            print("🗃️  N5 Output: Not created (fake_interest_points disabled or n5_output not provided)")
        print("="*60)
    
    def _run_fallback(self):
        """Fallback implementation if imports fail - completely self-contained."""
        print("Using self-contained fallback implementation...")
        
        try:
            # Load XML data
            print(f"Loading XML data from: {self.xml_input}")
            xml_data = self.load_xml_data(self.xml_input)
            if xml_data is None:
                print("Failed to load XML data")
                return False
            
            # Process target parameters
            target_image_size, target_overlap = self.process_target_parameters(
                self.target_image_size_string, self.target_overlap_string
            )
            if target_image_size is None or target_overlap is None:
                print("Failed to process target parameters")
                return False
            
            # Analyze dataset
            print("Analyzing dataset...")
            analysis_result = self.analyze_dataset(xml_data, target_image_size, target_overlap)
            if not analysis_result:
                print("Failed to analyze dataset")
                return False
            
            # Calculate adjusted parameters
            params = self.calculate_adjusted_parameters(analysis_result)
            if not params:
                print("Failed to calculate adjusted parameters")
                return False
            
            # Validate parameters
            if not self.validate_parameters(params):
                print("Parameter validation failed")
                return False
            
            # Perform image splitting
            if not self.perform_image_splitting(xml_data, params, analysis_result):
                print("Image splitting failed")
                return False
            
            # Save XML output
            print(f"Saving XML output to: {self.xml_output}")
            if not self.save_xml_output(xml_data, self.xml_output):
                print("Failed to save XML output")
                return False
            
            # Create N5 files if fake interest points are enabled
            if self.fake_interest_points:
                print("Creating fake interest points...")
                if not self.create_n5_files_parallel(xml_data, self.n5_output):
                    print("Failed to create N5 files")
                    return False
            
            print("✅ Fallback image splitting completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error in fallback image splitting: {e}")
            return False
