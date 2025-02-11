from bioio import BioImage
import bioio_tifffile
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

# This component loads tiff image data using bio-io library

class TiffImageReader:

    def __init__(self, dsxy, dsz, overlapping_interval, process_intervals, file_path):
        self.dsxy, self.dsz = dsxy, dsz
        self.overlapping_interval = overlapping_interval
        self.process_intervals = process_intervals
        self.file_path = file_path
        
        self.downsampled_dask_images = None
        self.image_data = {}

    # downsampling by factor of 2 for X,Y,Z
    def downsample(self, data, factor_dx, factor_dy, factor_dz, axes):
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
        
    # get image data and downsample
    def load_image_metadata(self, file_path):
        img = BioImage(file_path, reader=bioio_tifffile.Reader)
        initial_dask_images = img.get_dask_stack()[0, 0, 0, :, :, :]
        self.downsampled_dask_images = self.downsample(initial_dask_images, self.dsxy, self.dsxy, self.dsz, axes=[0, 1, 2])    

    # get image as a slice within bounds 
    def fetch_image_slice(self, z, interval):
        try: 
            lb, ub = interval['lower_bound'], interval['upper_bound']
            data_slice = self.downsampled_dask_images[z, lb[1]:ub[1], lb[2]:ub[2]].compute()
            return data_slice

        except Exception as e:
            print(f"Error fetching image slice: {e}")
            return
        
    # iterate through all intervals and then z values
    def fetch_all_slices(self):
        z_values = set()
        for interval in self.process_intervals:
            lower_z = interval['lower_bound'][2]
            upper_z = interval['upper_bound'][2]
            z_values.update(range(lower_z, upper_z + 1))
            for z in sorted(z_values):
                bounds_key = (z, tuple(interval['lower_bound']), tuple(interval['upper_bound']))
                self.image_data[bounds_key] = self.fetch_image_slice(z, interval)
                self.visualize_slice(self.image_data[bounds_key])

    # display image slice
    def visualize_slice(self, image):
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(image, cmap='gray')  
            plt.title(f"Image Slice")
            plt.colorbar()
            plt.show()
        except Exception as e:
            print(f"Error displaying slice: {e}")

    def run(self):
        self.load_image_metadata(self.file_path)
        self.fetch_all_slices()
            
        return self.image_data