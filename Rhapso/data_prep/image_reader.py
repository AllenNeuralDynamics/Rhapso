import zarr
import numpy as np
from bioio import BioImage
import bioio_tifffile
import dask.array as da
import zarr
import s3fs

# This class loads image data using predefined metadata of pathways

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
    
    def downsample(self, data, factor_dx, factor_dy, factor_dz, axes):
        """
        Downsamples a 3D array by block-wise mean pooling along specified axes.
        """
        for axis in axes:
            if axis == 0: 
                while factor_dz > 1:
                    data = da.coarsen(np.mean, data, {0:2}, trim_excess=True)
                    factor_dz //= 2  
            if axis == 1: 
                while factor_dx > 1:
                    data = da.coarsen(np.mean, data, {1:2}, trim_excess=True)
                    factor_dx //= 2
            if axis == 2:
                while factor_dy > 1:
                    data = da.coarsen(np.mean, data, {2:2}, trim_excess=True)
                    factor_dy //= 2
        return data 

    def fetch_image_data(self, record, dsxy, dsz):
        """
        Loads image chunk, downsamples it, and sub_chunks based on predefined intervals.
        """
        view_id = record['view_id']
        file_path = record['file_path']
        interval_key = record['interval_key']
        offset = record['offset']
        
        # Create image pathways using Dask
        if self.file_type == "tiff":
            img = CustomBioImage(file_path, reader=bioio_tifffile.Reader)
            dask_array = img.get_dask_stack()[0, 0, 0, :, :, :]
        
        elif self.file_type == "zarr":
            s3 = s3fs.S3FileSystem(anon=False)  
            full_path = f"{file_path}"
            store = s3fs.S3Map(root=full_path, s3=s3)
            zarr_array = zarr.open(store, mode='r')
            dask_array = da.from_zarr(zarr_array)[0, 0, :, :, :]

        # Downsample Dask array
        downsampled_stack = self.downsample(dask_array, dsxy, dsxy, dsz, axes=[0, 1, 2])

        # Get lower and upper bounds
        lb = list(interval_key[0])
        ub = list(interval_key[1])

        # Load image chunk into mem
        downsampled_image_chunk = downsampled_stack[lb[2]:ub[2], lb[1]:ub[1], lb[0]:ub[0]].compute()
    
        interval_key = (
            tuple(lb),
            tuple(ub),
            tuple((ub[0] - lb[0], ub[1] - lb[1], ub[2] - lb[2]))  
        )

        return view_id, interval_key, downsampled_image_chunk, offset, lb

    def run(self, metadata_df, dsxy, dsz):
            return self.fetch_image_data(metadata_df, dsxy, dsz)

