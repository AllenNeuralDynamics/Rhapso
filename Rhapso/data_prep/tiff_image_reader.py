from bioio import BioImage
import bioio_tifffile
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

# This component loads tiff image data using bio-io library

class TiffImageReader:
    def __init__(self, dsxy, dsz, process_intervals, file_path, view_id):
        self.dsxy, self.dsz = dsxy, dsz
        self.process_intervals = process_intervals
        self.file_path = file_path
        self.view_id = view_id
        self.shape = None
        self.downsampled_dask_images = None
        self.image_data = {}
        self.downsampled_slices = []

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

    def visualize_slice(self, image):
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(image, cmap='gray')  
            plt.title(f"Image Slice")
            plt.colorbar()
            plt.show()
        except Exception as e:
            print(f"Error displaying slice: {e}") 
    
    def load_and_process_slices(self, file_path):
        img = BioImage(file_path, reader=bioio_tifffile.Reader)
        full_dask_stack = img.get_dask_stack()[0, 0, 0, :, :, :]
        downsampled_stack = self.downsample(full_dask_stack, self.dsxy, self.dsxy, self.dsz, axes=[0, 1, 2])
        image_chunks = []

        # Process only the required slices
        for interval in self.process_intervals:
            z_start, z_stop = interval['lower_bound'][2], interval['upper_bound'][2] + 1
            y_start, y_stop = interval['lower_bound'][1], interval['upper_bound'][1] + 1
            x_start, x_stop = interval['lower_bound'][0], interval['upper_bound'][0] + 1

            downsampled_image_chunk = downsampled_stack[z_start:z_stop, y_start:y_stop, x_start:x_stop]
       
            interval_key = (
                tuple(interval['lower_bound']),
                tuple(interval['upper_bound']),
                tuple([x_stop - x_start, y_stop - y_start, z_stop - z_start])  
            )

            image_chunks.append({
                'view_id': self.view_id,
                'interval_key': interval_key,
                'image_chunk': downsampled_image_chunk
            })
            
        return image_chunks

    def run(self):
        return self.load_and_process_slices(self.file_path)