import numpy as np

"""
Metadata Builder constructs lists of pathways to each image chunk needed for interest point detection
"""

class MetadataBuilder:
    def __init__(self, dataframes, overlapping_area, image_file_prefix, file_type, dsxy, dsz, 
                 chunks_per_bound, sigma, run_type, level):
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
    
    def get_xyz_chunks(self, starts, stops):
        scale = np.asarray((self.dsxy, self.dsxy, self.dsz), dtype=np.float64)

        def split(chunk_lb, chunk_stop, num_chunks):
            if num_chunks == 1:
                return [(chunk_lb, chunk_stop)]

            extents = np.asarray(chunk_stop) - np.asarray(chunk_lb)
            available_axes = np.where(extents > 1)[0]

            if available_axes.size == 0:
                raise ValueError(
                    f"Cannot split bounds {chunk_lb} to {chunk_stop} "
                    f"into {num_chunks} non-empty chunks"
                )

            effective_extents = extents / scale
            axis = int(available_axes[np.argmax(effective_extents[available_axes])])

            left_count = num_chunks // 2
            right_count = num_chunks - left_count

            target = chunk_lb[axis] + extents[axis] * (left_count / num_chunks)
            cut = int(round(target / 128)) * 128
            cut = max(chunk_lb[axis] + 1, min(cut, chunk_stop[axis] - 1))

            left_stop = list(chunk_stop)
            left_stop[axis] = cut

            right_lb = list(chunk_lb)
            right_lb[axis] = cut

            return (
                split(chunk_lb, tuple(left_stop), left_count)
                + split(tuple(right_lb), chunk_stop, right_count)
            )

        total_voxels = np.prod(np.asarray(stops) - np.asarray(starts))

        if total_voxels < self.chunks_per_bound:
            raise ValueError(
                f"Cannot split bounds {starts} to {stops} into "
                f"{self.chunks_per_bound} non-empty chunks"
            )

        return split(starts, stops, self.chunks_per_bound)
    
    def build_image_metadata(self, process_intervals, file_path, view_id, split_min):
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
                span = tuple(int(ub_fixed[i] - lb_fixed[i] + 1) for i in range(3))
                interval_key = (lb_fixed, ub_fixed, span)

                self.metadata.append({
                    'view_id': view_id,
                    'file_path': file_path,
                    'interval_key': interval_key,
                    'offset': (0, 0, 0),
                    'lb': lb_fixed,
                    'split_min': split_min,
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
                        actual_ub = (x_stop - 1, y_stop - 1, z_start + z_end - 1)
                        span = tuple(actual_ub[i] - actual_lb[i] + 1 for i in range(3))

                        interval_key = (actual_lb, actual_ub, span)

                        self.metadata.append({
                            'view_id': view_id,
                            'file_path': file_path,
                            'interval_key': interval_key,
                            'offset': (0, 0, z),
                            'lb' : lb,
                            'split_min': split_min,
                        })  

                elif self.file_type == "zarr":
                    starts = tuple(int(v) for v in lb)
                    stops = tuple(int(v) + 1 for v in ub)

                    for core_lb, core_stop in self.get_xyz_chunks(starts, stops):
                        core_ub = tuple(core_stop[i] - 1 for i in range(3))

                        actual_lb = tuple(max(lb[i], core_lb[i] - self.overlap) for i in range(3))
                        actual_ub = tuple(min(ub[i], core_ub[i] + self.overlap) for i in range(3))
                        span = tuple(actual_ub[i] - actual_lb[i] + 1 for i in range(3))
                        offset = tuple(actual_lb[i] - lb[i] for i in range(3))

                        self.metadata.append({
                            'view_id': view_id,
                            'file_path': file_path,
                            'interval_key': (actual_lb, actual_ub, span),
                            'offset': offset,
                            'lb': lb,
                            'split_min': split_min,
                        })
    
    def build_paths(self):
        """
        Iterates through views to interface metadata building
        """
        for _, row in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['view_setup']}"
            process_intervals = self.overlapping_area[view_id]
            split_min = row.get("split_min")
            
            if self.file_type == 'zarr':
                file_path = self.image_file_prefix + row['file_path'] + f'/{self.level}'
            elif self.file_type == 'tiff':
                file_path = self.image_file_prefix + row['file_path'] 
            else:
                raise ValueError(f"Unsupported file_type: {self.file_type!r}")
            
            if self.run_type == 'ray':
                self.build_image_metadata(process_intervals, file_path, view_id, split_min)
            else:
                raise ValueError(f"Unsupported run type: {self.run_type!r}")

    def run(self):
        self.build_paths()
        print("Image Metadata Computed")

        return self.metadata