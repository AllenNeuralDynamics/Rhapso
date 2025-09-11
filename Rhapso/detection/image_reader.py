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
            store = s3fs.S3Map(root=full_path, s3=s3)
            zarr_array = zarr.open(store, mode='r')
            dask_array = da.from_zarr(zarr_array)[0, 0, :, :, :]

        dask_array = dask_array.astype(np.float32)
        dask_array = dask_array.transpose()

        # Downsample Dask array
        downsampled_stack = self.interface_downsampling(dask_array, dsxy, dsz)

        # Get lower and upper bounds
        lb = list(interval_key[0])
        ub = list(interval_key[1])

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

