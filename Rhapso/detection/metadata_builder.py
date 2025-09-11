import numpy as np

"""
Metadata Builder constructs lists of pathways to each image chunk needed for interest point detection
"""

class MetadataBuilder:
    def __init__(self, dataframes, overlapping_area, image_file_prefix, file_type, dsxy, dsz, chunks_per_bound, sigma, run_type,
                 level
        ):
        self.image_loader_df = dataframes['image_loader']
        self.overlapping_area = overlapping_area
        self.image_file_prefix = image_file_prefix
        self.file_type = file_type
        self.dsxy = dsxy
        self.dsz = dsz  
        self.chunks_per_bound = chunks_per_bound
        self.run_type = run_type
        self.level = level
        self.overlap = int(np.ceil(3 * sigma))
        self.sub_region_chunking = not chunks_per_bound == 0
        self.metadata = []
    
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

                    num_chunks = self.chunks_per_bound

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
                            'offset': z,
                            'lb' : lb
                        })  

                elif self.file_type == "zarr":

                    # # Compute cropped shape from bounds
                    x_start, y_start, z_start = lb
                    x_stop, y_stop, z_stop = [u + 1 for u in ub]

                    num_chunks = self.chunks_per_bound
                    
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
                            'offset': z,
                            'lb' : lb
                        })  
    
    def build_paths(self):
        """
        Iterates through views to interface metadata building
        """
        for _, row in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['view_setup']}"
            process_intervals = self.overlapping_area[view_id]
            
            if self.file_type == 'zarr':
                file_path = self.image_file_prefix + row['file_path'] + f'/{self.level}'
            elif self.file_type == 'tiff':
                file_path = self.image_file_prefix + row['file_path'] 
            else:
                raise ValueError(f"Unsupported file_type: {self.file_type!r}")
            
            if self.run_type == 'ray':
                self.build_image_metadata(process_intervals, file_path, view_id)
            else:
                raise ValueError(f"Unsupported run type: {self.run_type!r}")

    def run(self):
        self.build_paths()
        return self.metadata