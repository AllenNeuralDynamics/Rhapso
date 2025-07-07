from .dask_image_reader import DaskImageReader
import numpy as np
import zarr
import math
import zarr
import s3fs

# This class creates a list of metadata defining sub-chunk bounds

class MetadataBuilder:
    def __init__(self, dataframes, overlapping_area, prefix, file_type, dsz, dsxy, mem_per_worker, sigma, run_type):
        self.image_loader_df = dataframes['image_loader']
        self.overlapping_area = overlapping_area
        self.prefix = prefix
        self.file_type = file_type
        self.dsz = dsz
        self.metadata = []
        self.dsxy = dsxy
        self.sub_region_chunking = not mem_per_worker == 0  
        self.memory_per_worker = mem_per_worker
        self.run_type = run_type
        self.overlap = int(np.ceil(3 * sigma))
    
    def build_image_metadata(self, process_intervals, file_path, view_id):
        """
        Builds list of metadata with optional sub-chunking
        """
        for bound_set in process_intervals:
            lb = tuple(int(x) for x in bound_set['lower_bound'])
            ub = tuple(int(x) for x in bound_set['upper_bound'])

            # No chunking needed
            if not self.sub_region_chunking:
                lb_fixed = tuple(int(x) for x in lb)
                ub_fixed = tuple(int(x) for x in ub)
                span = tuple(int(ub_fixed[i] - lb_fixed[i]) for i in range(3))
                interval_key = (lb_fixed, ub_fixed, span)

                self.metadata.append({
                    'view_id': view_id,
                    'file_path': file_path,
                    'interval_key': interval_key,
                    'offset': 0,
                    'lb': lb_fixed
                }) 

            # Apply sub-region chunking
            else:       
                if self.file_type == "tiff":
                    # TODO - Add dynamic chunking strategy for large datasets
                    num_chunks = 3

                    # Compute cropped shape from bounds
                    x_start, y_start, z_start = lb
                    x_stop, y_stop, z_stop = [u + 1 for u in ub]
                    cropped_shape = (z_stop - z_start, y_stop - y_start, x_stop - x_start)

                    # Create num_chunks sets of z indices 
                    z_indices = np.array_split(np.arange(cropped_shape[0]), num_chunks)

                    for chunk in z_indices:
                        z = max(0, chunk[0] - self.overlap)
                        z_end = min(chunk[-1] + 1 + self.overlap, cropped_shape[0])

                        actual_lb = (x_start, y_start, z_start + z)
                        actual_ub = (x_stop, y_stop, z_start + z_end)

                        span = tuple(actual_ub[i] - actual_lb[i] for i in range(3))
                        interval_key = (actual_lb, actual_ub, span)

                        self.metadata.append({
                            'view_id': view_id,
                            'file_path': file_path,
                            'interval_key': interval_key,
                            'offset': z_start,
                            'lb' : lb
                        })  

                elif self.file_type == "zarr":
                    s3 = s3fs.S3FileSystem(anon=False)  
                    store = s3fs.S3Map(root=f"{file_path}", s3=s3)
                    zarr_array = zarr.open(store, mode='r')

                    # Compute cropped shape from bounds
                    x_start, y_start, z_start = lb
                    x_stop, y_stop, z_stop = [u + 1 for u in ub]

                    cropped_shape = (z_stop - z_start, y_stop - y_start, x_stop - x_start)

                    # Calculate estimated memory size
                    voxel_bytes = np.dtype(zarr_array.dtype).itemsize
                    cropped_voxels = np.prod(cropped_shape)
                    cropped_total_bytes = cropped_voxels * voxel_bytes

                    # Target memory per chunk
                    target_chunk_mem = int(self.memory_per_worker * 0.04)  
                    num_chunks = max(1, math.ceil(cropped_total_bytes / target_chunk_mem))
                    
                    # Create num_chunks sets of z indices 
                    z_indices = np.array_split(np.arange(z_stop - z_start), num_chunks)
                    
                    for chunk in z_indices:
                        z = max(0, chunk[0] - self.overlap)
                        z_end = min(chunk[-1] + 1 + self.overlap, z_stop - z_start)

                        actual_lb = (lb[0], lb[1], z_start + z)        
                        actual_ub = (ub[0], ub[1], z_start + z_end)

                        span = tuple(actual_ub[i] - actual_lb[i] for i in range(3))
                        interval_key = (actual_lb, actual_ub, span)

                        self.metadata.append({
                            'view_id': view_id,
                            'file_path': file_path,
                            'interval_key': interval_key,
                            'offset': z_start,
                            'lb' : lb
                        })  
    
    def build_image_metadata_dask(self, process_intervals, file_path, view_id):
        """
        Loads image data based on the file type specified and processes it using the corresponding image reader.
        """         
        image_reader = DaskImageReader(self.file_type, self.dsxy, self.dsz, process_intervals, file_path, view_id)                       
        self.metadata.extend(image_reader.run())
    
    def build_paths(self):
        """
        Iterates through views to interface metadata building
        """
        for _, row in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['view_setup']}"
            process_intervals = self.overlapping_area[view_id]
            if self.file_type == 'zarr':
                file_path = self.prefix + row['file_path'] + f'/{0}'
            elif self.file_type == 'tiff':
                file_path = self.prefix + row['file_path'] 
            
            if self.run_type == 'dask':
                self.build_image_metadata_dask(process_intervals, file_path, view_id)
            else:
                self.build_image_metadata(process_intervals, file_path, view_id)

    def run(self):
        self.build_paths()
        return self.metadata