from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
import numpy as np
from scipy.ndimage import map_coordinates

# This component implements the difference of gaussian algorithm on image chunks
# We need to return output of: array [[[viewID],[Interval], [interestpoints],[intensities]]]

class DifferenceOfGaussian:
    def __init__(self):
        self.min_intensity = 0.0
        self.max_intensity = 2048.0
        self.sigma = 1.8
        self.threshold = 0.008
    
    def gaussian_3d(self, xyz, amplitude, xo, yo, zo, sigma_x, sigma_y, sigma_z, offset):
        x, y, z = xyz
        g = offset + amplitude * np.exp(
            -(((x - xo) ** 2) / (2 * sigma_x ** 2) +
            ((y - yo) ** 2) / (2 * sigma_y ** 2) +
            ((z - zo) ** 2) / (2 * sigma_z ** 2)))
        
        return g.ravel()

    def refine_peaks(self, peaks, image):
        if peaks is None:
            return []
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

    def compute_difference_of_gaussian(self, image):
        shape = 3
        initial_sigma = self.sigma
        min_peak_value = self.threshold
        min_initial_peak_value = min_peak_value / 3.0

        # normalize image using min/max intensities
        input_float = self.normalize_image(image)                                              

        # calculate gaussian blur levels 
        sigma_1, sigma_2 = self.compute_sigmas(initial_sigma, shape) 
        # print("we just got sigmas", sigma_1 + sigma_2)                           

        # apply gaussian blur
        blurred_image_1 = self.apply_gaussian_blur(input_float, sigma_1, shape)                
        blurred_image_2 = self.apply_gaussian_blur(input_float, sigma_2, shape)

        # subtract blurred images
        dog = blurred_image_1 - blurred_image_2

        # detect peaks
        peaks = peak_local_max(dog, threshold_rel=min_initial_peak_value)

        # print("we just got peaks:", peaks)

        # refine localization 
        final_peaks = self.refine_peaks(peaks, image) 

        # print("we just got final peaks: ", final_peaks)
         
        return final_peaks 
    
    def interpolation(self, image, interest_points):
        if interest_points is None or len(interest_points) == 0:
            return np.array([]) 

        points = np.array(interest_points).T
        intensities = map_coordinates(image, points, order=1, mode='nearest')
        return intensities
    
    def upsample_coordinates(self, points, dsxy, dsz):
        if points is None:
            return []
        return [(point[0] * dsxy, point[1] * dsxy, point[2] * dsz) for point in points]
    
    def run(self, image_chunk, dsxy, dsz):
        final_peaks = self.compute_difference_of_gaussian(image_chunk)
        # print("we just got final peaks")
        intensities = self.interpolation(image_chunk, final_peaks)
        # print("our new intensities", intensities)
        upsampled_final_peaks = self.upsample_coordinates(final_peaks, dsxy, dsz)

        return {
            'interest_points': upsampled_final_peaks,
            'intensities': intensities
        }