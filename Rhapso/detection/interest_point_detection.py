from data_preparation.tiff_image_reader import TiffImageReader  # python version wants this way
from data_preparation.zarr_image_reader import ZarrImageReader  # 
# from ..data_preparation.zarr_image_reader import ZarrImageReader
# from ..data_preparation.tiff_image_reader import TiffImageReader
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from dask import delayed
from dask import compute
import numpy as np
import math

# This component detects interest points within overlapping areas of images

class PythonInterestPointDetection:
    def __init__(self, dataframes, overlapping_area, dsxy, dsz, prefix, file_type, image_bucket_name):
        self.dataframes = dataframes
        self.image_loader_df = dataframes['image_loader']
        self.overlapping_area = overlapping_area
        self.dsxy = dsxy
        self.dsz = dsz
        self.prefix = prefix
        self.file_type = file_type
        self.image_bucket_name = image_bucket_name
        
        self.localization = 1
        self.downsample_z = 2
        self.downsample_xy = 4
        self.image_sigma_x = 0.5
        self.image_sigma_y = 0.5
        self.image_sigma_z = 0.5
        self.min_intensity = 0.0
        self.max_intensity = 2048.0
        self.block_size = 1024,1024,1024
        self.sigma = 1.8
        self.threshold = 0.008
        self.find_min = False
        self.find_max = True
        self.image_data = None
        self.k_min_1_inv = None
        self.mask_float = None
        self.interest_points = []
    
    def gaussian_3d(self, xyz, amplitude, xo, yo, zo, sigma_x, sigma_y, sigma_z, offset):
        x, y, z = xyz
        g = offset + amplitude * np.exp(
            -(((x - xo) ** 2) / (2 * sigma_x ** 2) +
            ((y - yo) ** 2) / (2 * sigma_y ** 2) +
            ((z - zo) ** 2) / (2 * sigma_z ** 2)))
        
        return g.ravel()

    def refine_peaks(self, peaks, image):
        refined_peaks = []
        window_size = 5

        for peak in peaks:
            x, y, z = peak
            # Check if the peak is too close to any border
            if (x < window_size or x >= image.shape[0] - window_size or
                y < window_size or y >= image.shape[1] - window_size or
                z < window_size or z >= image.shape[2] - window_size):
                continue

            # Extract a volume around the peak
            patch = image[x-window_size:x+window_size+1,
                        y-window_size:y+window_size+1,
                        z-window_size:z+window_size+1]

            # Prepare the data for fitting
            x_grid, y_grid, z_grid = np.mgrid[
                -window_size:window_size+1,
                -window_size:window_size+1,
                -window_size:window_size+1]
            initial_guess = (patch.max(), 0, 0, 0, 1, 1, 1, 0)  # Amplitude, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset

            try:
                popt, _ = curve_fit(
                    self.gaussian_3d, (x_grid, y_grid, z_grid), patch.ravel(),
                    p0=initial_guess)
                refined_x = x + popt[1]
                refined_y = y + popt[2]
                refined_z = z + popt[3]
                refined_peaks.append((refined_x, refined_y, refined_z))
            except Exception as e:
                refined_peaks.append((x, y, z))

        return refined_peaks
    
    def apply_gaussian_blur(self, input_float, sigma, shape):
        blurred_image = input_float
        
        for i in range(shape):
            blurred_image = gaussian_filter(blurred_image, sigma=sigma[i], mode='reflect')
        
        return blurred_image
    
    def compute_sigma(self, steps, k, initial_sigma):
        sigma = np.zeros(steps + 1)
        sigma[0] = initial_sigma

        for i in range(1, steps + 1):
            sigma[i] = sigma[i - 1] * k

        return sigma

    def compute_sigma_difference(self, sigma, image_sigma):
        steps = len(sigma) - 1
        sigma_diff = np.zeros(steps + 1)
        sigma_diff[0] = np.sqrt(sigma[0]**2 - image_sigma**2)

        for i in range(1, steps + 1):
            sigma_diff[i] = np.sqrt(sigma[i]**2 - image_sigma**2)

        return sigma_diff
    
    def compute_sigmas(self, initial_sigma, shape):
        k = 2 ** (1 / 4)
        self.k_min_1_inv = 1.0 / (k - 1.0)
        steps = 3
        sigma = np.zeros((2, shape))

        for i in range(shape):
            sigma_steps_x = self.compute_sigma(steps, k, initial_sigma)
            sigma_steps_diff_x = self.compute_sigma_difference(sigma_steps_x, 0.5)
            sigma[0][i] = sigma_steps_diff_x[0]  
            sigma[1][i] = sigma_steps_diff_x[1]
        
        return sigma
    
    def normalize_image(self, image):
        normalized_image = (image - self.min_intensity) / (self.max_intensity - self.min_intensity)
        return normalized_image

    def compute_difference_of_gaussian(self, image, shape):
        initial_sigma = self.sigma
        min_peak_value = self.threshold
        min_initial_peak_value = min_peak_value / 3.0

        # normalize image using min/max intensities
        input_float = self.normalize_image(image)                                               

        # calculate gaussian blur levels 
        sigma_1, sigma_2 = self.compute_sigmas(initial_sigma, shape)                             

        # apply gaussian blur
        blurred_image_1 = self.apply_gaussian_blur(input_float, sigma_1, shape)                
        blurred_image_2 = self.apply_gaussian_blur(input_float, sigma_2, shape)

        # subtract blurred images
        dog = blurred_image_1 - blurred_image_2

        # detect peaks
        peaks = peak_local_max(dog, threshold_rel=min_initial_peak_value)

        # refine localization 
        final_peaks = self.refine_peaks(peaks, image)   
         
        return final_peaks 

    def apply_dog_to_all_slices(self, downsampled_slices, shape):

        # This function does the heavy lifting of this component (bottleneck)
        
        # PYTHON MAP FUNCTION RUN TIME - 20 MIN
        # results = map(lambda slice: self.compute_difference_of_gaussian(slice, shape), downsampled_slices)
        # return list(results)
    
        # PYTHON BRUTE FORCE - RUN TIME 22 MIN
        # results = []  
        # for slice_dask in downsampled_slices:
        #     dog_result = self.compute_difference_of_gaussian(slice_dask, shape)
        #     results.append(dog_result) 
        # return results     

        # DASK MAP FUNCTION - RUN TIME 2 MIN
        results = []  
        for slice_dask in downsampled_slices:
            print(slice_dask)
            print(shape)
            dog_result = delayed(self.compute_difference_of_gaussian)(slice_dask, shape)
            results.append(dog_result) 
        return compute(*results)                         
    
    def load_image_data(self, process_intervals, file_path):
        if self.file_type == 'zarr':
            image_reader = ZarrImageReader(self.dsxy, self.dsz, process_intervals, file_path)
        elif self.file_type == 'tiff':
            image_reader = TiffImageReader(self.dsxy, self.dsz, process_intervals, file_path) 
                                      
        return image_reader.run()

    def interest_point_detection(self):
        interest_points = {}
        for _, row in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['view_setup']}"
            process_intervals = self.overlapping_area[view_id]
            if self.file_type == 'zarr':
                file_path = self.prefix + row['file_path'] + f'/{4}'
            elif self.file_type == 'tiff':
                file_path = self.prefix + row['file_path'] 
            image, shape = self.load_image_data(process_intervals, file_path)
            interest_points[view_id] = self.apply_dog_to_all_slices(image, len(shape))

        print(interest_points)

    def run(self):
        self.interest_point_detection()
    

    # def square(self, x):
    #     return x * x
    
    # def smooth_edge(self, kernel):
    #     L = len(kernel)
    #     slope = float('inf') 
    #     r = L
        
    #     while r > L // 2:
    #         r -= 1
    #         a = kernel[r] / self.square(L - r)
    #         if a < slope:
    #             slope = a
    #         else:
    #             r += 1
    #             break

    #     for x in range(r + 1, L):
    #         kernel[x] = slope * self.square(L - x)
        
    #     return kernel

    # def multiply(self, values, factor):   
    #     for x in range(len(values)):
    #         values[x] *= factor

    # def normalize_half_kernel(self, kernel):
    #     sum_val = 0.5 * kernel[0]
        
    #     for x in range(1, len(kernel)):
    #         sum_val += kernel[x]
        
    #     sum_val *= 2
    #     self.multiply(kernel, 1 / sum_val)
        
    #     return kernel
    
    # def half_kernel(self, sigma, size):
    #     two_sq_sigma = 2 * self.square(sigma)
    #     kernel = [0] * size

    #     kernel[0] = 1
    #     for x in range(1, size):
    #         kernel[x] = math.exp(-self.square(x) / two_sq_sigma)

    #     smooth_kernel = self.smooth_edge(kernel)
    #     normalized_smooth_kernel = self.normalize_half_kernel(smooth_kernel)

    #     return normalized_smooth_kernel
    
    # def half_kernels(self, sigma, shape):
    #     kernel = {}
    #     for i in range(shape):
    #         half_kernel_size = max(2, int(3 * sigma[i] + 0.5) + 1)
    #         kernel[i] = self.half_kernel(sigma[i], half_kernel_size)
        
    #     return kernel
    
    # def apply_gaussian_blur(self, input_float, sigma, shape):
    #     half_kernels = self.half_kernels(sigma, shape)                    # confirmed correct results

    #     full_kernels = {dim: self.mirror_kernel(half_kernels[dim]) for dim in half_kernels}

    #     for i, kernel in full_kernels.items():
    #         # Apply 1D convolution along each axis separately
    #         input_float = gaussian_filter(input_float, sigma=sigma[i], axis=i, mode='reflect')
        
    #     return input_float
