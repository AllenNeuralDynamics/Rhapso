import zarr
import s3fs
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

# This class loads Zarr image data using Zarr library

class ZarrImageReader():
    def __init__(self, dsxy, dsz, process_intervals, file_path, view_id):
        self.dsx = dsxy
        self.dsy = dsxy
        self.dsz = dsz
        self.process_intervals = process_intervals
        self.file_path = file_path
        self.view_id = view_id

    def display_images(self, dask_array):
        """
        DEV ONLY - helper function to view the image data per slice
        """
        for i, img in enumerate(dask_array):
            img = img.compute_chunk_sizes()
            delayed_chunks = img.to_delayed().ravel() 
            for j, delayed_chunk in enumerate(delayed_chunks):
                chunk = delayed_chunk.compute()  
                num_z = chunk.shape[2]  
                for z in range(num_z):
                    z_slice = chunk[0, 0, z, :, :]
                    plt.figure(figsize=(10, 8), dpi=100)
                    im = plt.imshow(z_slice, cmap='gray', interpolation='bessel')
                    plt.title(f"Chunk {j+1}, Z-Slice {z+1}")
                    plt.colorbar(im)  
                    plt.show()

    def downsample(self, data, factor_dx, factor_dy, factor_dz, axes):
        """
        Downsamples the data array by averaging over dimensions specified by axes, taking into consideration the 
        maximum factor that divides the dimension size.
        """

        # Map each axis to its respective downsampling factor
        factors = {0: factor_dz, 1: factor_dx, 2: factor_dy}

        # Iterate through each specified axis to apply downsampling
        for axis in axes:
            original_size = data.shape[axis]
            
            while factors[axis] > 1:
                # Calculate the highest possible factor that divides the dimension size
                possible_factor = 2
                while possible_factor <= factors[axis] and original_size % possible_factor == 0:
                    possible_factor *= 2
                
                 # Step back to the last valid factor
                possible_factor //= 2 

                # If a valid factor greater than 1 is found, apply coarsening
                if possible_factor > 1:
                    data = da.coarsen(np.mean, data, {axis: possible_factor}, trim_excess=True)
                    original_size //= possible_factor
                    factors[axis] //= possible_factor
                else:
                    break  

        return data

    def fetch_zarr_data(self):
        """
        Retrieves and processes Zarr-formatted image data from an S3 bucket, downsamples it, and segments it into 
        predefined intervals for further processing.
        """

        # Initialize the connection to S3 with the specified bucket and path
        s3 = s3fs.S3FileSystem(anon=False)  
        full_path = f"{self.file_path}"
        store = s3fs.S3Map(root=full_path, s3=s3)

        # Open the Zarr store and convert it to a Dask array for lazy loading
        zarr_array = zarr.open(store, mode='r')
        dask_array = da.from_zarr(zarr_array)   

        # Downsample the Dask array using predefined factors along specified axes
        downsampled_stack = self.downsample(dask_array, self.dsx, self.dsy, self.dsz, axes=[0, 1, 2])
        image_chunks = []

        # Process each interval defined in process_intervals
        for bound_set in self.process_intervals:
            lb = bound_set['lower_bound']
            ub = bound_set['upper_bound']
            
            downsampled_image_chunk = downsampled_stack[0, 0, lb[2]:ub[2], lb[1]:ub[1], lb[0]:ub[0]]

            # Print image chunk size - dev helper function 
            # chunk_size_bytes = downsampled_image_chunk.size * downsampled_image_chunk.dtype.itemsize
            # chunk_size_mb = chunk_size_bytes / (1024 ** 2)
            # print(chunk_size_mb)
            
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
        return self.fetch_zarr_data()