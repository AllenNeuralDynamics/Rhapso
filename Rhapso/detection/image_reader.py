import zarr
import numpy as np
from bioio import BioImage
import bioio_tifffile
import dask.array as da
import s3fs

"""
Image Reader loads and downsamples Zarr and TIFF OME data
"""

class CustomBioImage(BioImage):
    def standard_metadata(self):
        pass
    
    def scale(self):
        pass
    
    def time_interval(self):
        pass

class ImageReader:
    def __init__(self, file_type):
        self.file_type = file_type

    def downsample(self, arr, axis):
        """
        Reduce size by 2 along `axis` by averaging adjacent elements
        """
        s0 = [slice(None)] * arr.ndim
        s1 = [slice(None)] * arr.ndim
        s0[axis] = slice(0, None, 2)
        s1[axis] = slice(1, None, 2)

        a0 = arr[tuple(s0)]
        a1 = arr[tuple(s1)]

        len1 = a1.shape[axis]
        s0c = [slice(None)] * a0.ndim
        s0c[axis] = slice(0, len1)
        a0 = a0[tuple(s0c)]

        return (a0 + a1) * 0.5

    def interface_downsampling(self, data, dsxy, dsz):
        """
        Downsample a 3D volume by powers of two by repeatedly halving along each axis
        """
        # Process X dimension
        f = dsxy
        while f > 1:
            data = self.downsample(data, axis=0)  
            f //= 2
        
        # Process Y dimension
        f = dsxy
        while f > 1:
            data = self.downsample(data, axis=1)  
            f //= 2
        
        # Process Z dimension
        f = dsz
        while f > 1:
            data = self.downsample(data, axis=2)  
            f //= 2
        
        return data

    def fetch_image_data(self, record, dsxy, dsz):
        """
        Loads image chunk, downsamples it, and sub_chunks based on predefined intervals.
        """
        view_id = record['view_id']
        file_path = record['file_path']
        interval_key = record['interval_key']
        offset = record['offset']
        lower_bound = record['lb']
        
        # Create image pathways using Dask
        if self.file_type == "tiff":
            img = CustomBioImage(file_path, reader=bioio_tifffile.Reader)
            dask_array = img.get_dask_stack()[0, 0, 0, :, :, :]
        
        elif self.file_type == "zarr":
            s3 = s3fs.S3FileSystem(anon=False)
            full_path = f"{file_path}"
            try:
                store = s3fs.S3Map(root=full_path, s3=s3)
                zarr_array = zarr.open(store, mode='r')
                dask_array = da.from_zarr(zarr_array)[0, 0, :, :, :]
            except Exception as e:
                print(f"[ImageReader] ERROR opening zarr at {full_path}: {e}")
                # Try to inspect root to show available multiscales
                try:
                    root_path = full_path.rsplit('/', 1)[0]
                    print(f"[ImageReader] Attempting to inspect root zarr at: {root_path}")
                    root_store = s3fs.S3Map(root=root_path, s3=s3)
                    root_zarr = zarr.open(root_store, mode='r')
                    available_levels = list(root_zarr.keys()) if hasattr(root_zarr, 'keys') else 'unknown'
                    print(f"[ImageReader] Available multiscale levels at root: {available_levels}")
                except Exception as e2:
                    print(f"[ImageReader] Could not inspect root zarr: {e2}")
                raise

        dask_array = dask_array.astype(np.float32)
        dask_array = dask_array.transpose()

        # Store original crop bounds (in level-0 coordinates) for later application
        crop_min = record.get('crop_min')
        crop_max = record.get('crop_max')

        # Downsample Dask array
        downsampled_stack = self.interface_downsampling(dask_array, dsxy, dsz)

        # Get lower and upper bounds
        lb = list(interval_key[0])
        ub = list(interval_key[1])

        # Bounds are in full-resolution (level 0) coordinates.
        # We loaded from a potentially downsampled multiscale level,
        # so we need to scale the bounds down by 2^level.
        # Extract level from file_path (last component after final /)
        try:
            level_str = file_path.rstrip('/').split('/')[-1]
            level = int(level_str)
            print(f"[ImageReader] file_path={file_path}, extracted level={level}")
            print(f"[ImageReader] Before scaling: lb={lb}, ub={ub}, downsampled_stack.shape={downsampled_stack.shape}")
            if level > 0:
                scale = 2 ** level
                lb = [x // scale for x in lb]
                ub = [x // scale for x in ub]
                print(f"[ImageReader] After scaling by 2^{level}={scale}: lb={lb}, ub={ub}")
        except (ValueError, IndexError) as e:
            print(f"[ImageReader] Level extraction failed ({e}); using bounds as-is")
            pass  # Level extraction failed; use bounds as-is

        # Now apply split tile crop if present (using scaled crop bounds)
        if crop_min is not None and crop_max is not None:
            if len(crop_min) != 3 or len(crop_max) != 3:
                raise ValueError(
                    f"crop_min and crop_max must both be length 3 for 3D cropping; "
                    f"got crop_min={crop_min}, crop_max={crop_max}"
                )

            # Scale crop bounds from level-0 coordinates to downsampled array
            # coordinates.  The array has been downsampled by 2^level (zarr
            # pyramid) AND by dsxy/dsz (interface_downsampling), so crop
            # bounds must be divided by the total factor.
            try:
                level_str = file_path.rstrip('/').split('/')[-1]
                level = int(level_str)
                total_scale_xy = (2 ** level) * dsxy
                total_scale_z = (2 ** level) * dsz
            except (ValueError, IndexError):
                total_scale_xy = dsxy
                total_scale_z = dsz

            # crop bounds are in XYZ order: [0]=X, [1]=Y use xy scale; [2]=Z
            scales = [total_scale_xy, total_scale_xy, total_scale_z]
            crop_min_scaled = [int(x // s) for x, s in zip(crop_min, scales)]
            crop_max_scaled = [int(np.ceil((x + 1) / s) - 1) for x, s in zip(crop_max, scales)]

            # Validate and clamp crop bounds to downsampled array dimensions
            array_shape = downsampled_stack.shape
            for i in range(3):
                if crop_min_scaled[i] < 0:
                    raise ValueError(
                        f"crop_min_scaled[{i}]={crop_min_scaled[i]} is negative"
                    )
                # Clamp crop_max to valid range
                crop_max_scaled[i] = min(crop_max_scaled[i], array_shape[i] - 1)
                if crop_min_scaled[i] > crop_max_scaled[i]:
                    raise ValueError(
                        f"crop_min_scaled[{i}]={crop_min_scaled[i]} > crop_max_scaled[{i}]={crop_max_scaled[i]}"
                    )

            print(f"[ImageReader] Applying crop: crop_min_scaled={crop_min_scaled}, crop_max_scaled={crop_max_scaled}")
            downsampled_stack = downsampled_stack[
                crop_min_scaled[0]:crop_max_scaled[0] + 1,
                crop_min_scaled[1]:crop_max_scaled[1] + 1,
                crop_min_scaled[2]:crop_max_scaled[2] + 1
            ]

        # Load image chunk into mem
        downsampled_image_chunk = downsampled_stack[lb[0]:ub[0]+1, lb[1]:ub[1]+1, lb[2]:ub[2]+1].compute()
    
        interval_key = (
            tuple(lb),
            tuple(ub),
            tuple((ub[0] - lb[0]+1, ub[1] - lb[1]+1, ub[2] - lb[2]+1))  
        )

        return view_id, interval_key, downsampled_image_chunk, offset, lower_bound

    def run(self, metadata_df, dsxy, dsz):
        """
        Executes the entry point of the script.
        """
        return self.fetch_image_data(metadata_df, dsxy, dsz)

