from bioio import BioImage
import bioio_tifffile
import numpy as np
import dask.array as da
import zarr
import s3fs

# This class loads image data using Dask

class CustomBioImage(BioImage):
    def standard_metadata(self):
        pass
    
    def scale(self):
        pass
    
    def time_interval(self):
        pass

class DaskImageReader:
    def __init__(self, file_type, dsxy, dsz, process_intervals, file_path, view_id):
        self.file_type = file_type
        self.dsxy, self.dsz = dsxy, dsz
        self.process_intervals = process_intervals
        self.file_path = file_path
        self.view_id = view_id
        self.shape = None
        self.downsampled_dask_images = None
        self.image_data = {}
        self.downsampled_slices = []

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
    
    def fetch_image_data(self):
        """
        Loads image chunk, downsamples it, and segments into sub-chunks based on predefined intervals.
        """
        # Fetch image data as dask array
        if self.file_type == "tiff":
            img = BioImage(self.file_path, reader=bioio_tifffile.Reader)
            dask_array = img.get_dask_stack()[0, 0, 0, :, :, :]
        
        elif self.file_type == "zarr":
            s3 = s3fs.S3FileSystem(anon=False)  
            full_path = f"{self.file_path}"
            store = s3fs.S3Map(root=full_path, s3=s3)
            zarr_array = zarr.open(store, mode='r')
            dask_array = da.from_zarr(zarr_array)[0, 0, :, :, :]

        # Downsample the Dask array using factors along specified axes
        downsampled_stack = self.downsample(dask_array, self.dsxy, self.dsxy, self.dsz, axes=[0, 1, 2])
        image_chunks = []

        # Process each interval defined in process_intervals
        for bound_set in self.process_intervals:
            lb = bound_set['lower_bound']
            ub = bound_set['upper_bound']
            
            # Fetch image chunk
            downsampled_image_chunk = downsampled_stack[lb[2]:ub[2], lb[1]:ub[1], lb[0]:ub[0]]
            
            interval_key = (
                tuple(lb),
                tuple(ub),
                tuple((ub[0] - lb[0], ub[1] - lb[1], ub[2] - lb[2])) 
            )

            image_chunks.append({
                'view_id': self.view_id,
                'interval_key': interval_key,
                'image_chunk': downsampled_image_chunk
            })
        
        return image_chunks

    def run(self):
        """
        Executes the entry point of the script.
        """
        return self.fetch_image_data()