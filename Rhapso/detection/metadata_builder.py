import os
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
    
    def build_image_metadata(self, process_intervals, file_path, view_id, crop_min=None, crop_max=None):
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
                    'lb': lb_fixed,
                    'crop_min': crop_min,
                    'crop_max': crop_max
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
                            'lb' : lb,
                            'crop_min': crop_min,
                            'crop_max': crop_max
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
                            'lb' : lb,
                            'crop_min': crop_min,
                            'crop_max': crop_max
                        })

    def build_paths(self):
        """
        Iterates through views to interface metadata building
        """
        is_split = 'crop_min' in self.image_loader_df.columns

        for _, row in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['view_setup']}"
            process_intervals = self.overlapping_area[view_id]

            if self.file_type == 'zarr':
                if is_split:
                    # zarr_base_path is the root (e.g., SPIM.ome.zarr/),
                    # file_path has the per-tile name (e.g., Tile_X_..._ch_405.zarr).
                    # Multiscale levels live inside each tile zarr.
                    file_path = os.path.join(row['zarr_base_path'], row['file_path'])
                    print(f"[MetadataBuilder] split=True, zarr_base_path={row['zarr_base_path']}, per_tile={row['file_path']}, joined={file_path}")
                else:
                    file_path = self.image_file_prefix
                    print(f"[MetadataBuilder] split=False, using image_file_prefix={file_path}")
                # Append multiscale level if not already present
                if self.level is not None and not str(file_path).rstrip('/').endswith(str(self.level)):
                    file_path = os.path.join(file_path, str(self.level))
                    print(f"[MetadataBuilder] Appended level={self.level}, final path={file_path}")
            elif self.file_type == 'tiff':
                file_path = os.path.join(self.image_file_prefix, row['file_path'])
            else:
                raise ValueError(f"Unsupported file_type: {self.file_type!r}")

            # Extract and scale crop bounds for split tiles
            crop_min = None
            crop_max = None
            if is_split:
                scale = 2 ** self.level if self.level is not None else 1
                cmin = [int(v) // scale for v in row['crop_min'].split()]
                # For inclusive bounds, use a ceil-style mapping for crop_max to avoid shrinking coverage
                cmax = [int(np.ceil((int(v) + 1) / scale) - 1) for v in row['crop_max'].split()]
                crop_min = cmin
                crop_max = cmax

            if self.run_type == 'ray':
                self.build_image_metadata(process_intervals, file_path, view_id, crop_min, crop_max)
            else:
                raise ValueError(f"Unsupported run type: {self.run_type!r}")

    def run(self):
        self.build_paths()
        return self.metadata