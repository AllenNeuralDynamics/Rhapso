import zarr
import s3fs
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

class ZarrImageReader():
    def __init__(self, dsxy, dsz, process_intervals, file_path):
        self.dsx = dsxy
        self.dsy = dsxy
        self.dsz = dsz
        self.process_intervals = process_intervals
        self.file_path = file_path

    def display_images(self, dask_array):
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

    def downsample(self, image_list, factor_dx, factor_dy, factor_dz):
        downsampled_images = []
        for data in image_list:
            while factor_dz > 1:
                data = da.coarsen(np.mean, data, {0: 2}, trim_excess=True)
                factor_dz //= 2
            while factor_dy > 1:
                data = da.coarsen(np.mean, data, {2: 2}, trim_excess=True)
                factor_dy //= 2
            while factor_dx > 1:
                data = da.coarsen(np.mean, data, {1: 2}, trim_excess=True)
                factor_dx //= 2
            downsampled_images.append(data)
        
        return downsampled_images

    def fetch_zarr_data(self):
        s3 = s3fs.S3FileSystem(anon=False)  
        full_path = f"{self.file_path}"
        store = s3fs.S3Map(root=full_path, s3=s3)
        zarr_array = zarr.open(store, mode='r')
        dask_array = da.from_zarr(zarr_array)

        extracted_images = []
        for bound_set in self.process_intervals:
            lb = bound_set['lower_bound']
            ub = bound_set['upper_bound']
            extracted_image = dask_array[:, :, lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]
            extracted_images.append(extracted_image)
        
        return extracted_images

    def run(self):
        zarr_data = self.fetch_zarr_data()
        return self.downsample(zarr_data, self.dsx, self.dsy, self.dsz)
    
