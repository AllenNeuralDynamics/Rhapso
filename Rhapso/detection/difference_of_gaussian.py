from scipy.ndimage import gaussian_filter 
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
import numpy as np
from scipy.ndimage import map_coordinates

# This class implements the difference of gaussian algorithm on image chunks

class DifferenceOfGaussian:
    def __init__(self, min_intensity, max_intensity, sigma, threshold):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.sigma = sigma
        self.threshold = threshold
    
    def gaussian_3d(self, xyz, amplitude, zo, yo, xo, sigma_x, sigma_y, sigma_z, offset):
        """
        Computes the 3D Gaussian value for given coordinates and Gaussian parameters.
        """
        x, y, z = xyz
        g = offset + amplitude * np.exp(
            -(((x - xo) ** 2) / (2 * sigma_x ** 2) +
            ((y - yo) ** 2) / (2 * sigma_y ** 2) +
            ((z - zo) ** 2) / (2 * sigma_z ** 2)))
        
        return g.ravel()

    def refine_peaks(self, peaks, image):
        """
        Refines the position of detected peaks in a 3D image by fitting a Gaussian model to local
        neighborhoods around each peak. Peaks too close to the image borders are excluded from refinement.
        """
        if peaks is None: return []
        
        refined_peaks = []
        window_size = 5

        # Create 3D grids for the neighborhood around a point, spanning the defined window size in all directions
        z_grid, y_grid, x_grid = np.mgrid[
            -window_size:window_size+1,
            -window_size:window_size+1,
            -window_size:window_size+1]

        for peak in peaks:
            z, y, x = peak

            # Check if the peak is too close to any border
            if (z < window_size or z >= image.shape[0] - window_size or
                y < window_size or y >= image.shape[1] - window_size or
                x < window_size or x >= image.shape[2] - window_size):
                continue

            # Extract a volume around the peak
            patch = image[z-window_size:z+window_size+1,
                      y-window_size:y+window_size+1,
                      x-window_size:x+window_size+1]
            
            # Prepare the data for fitting
            initial_guess = (patch.max(), 0, 0, 0, 1, 1, 1, 0)  # Amplitude, z0, y0, x0, sigma_z, sigma_y, sigma_x, offset
            try:
                popt, _ = curve_fit(
                    self.gaussian_3d, (z_grid, y_grid, x_grid), patch.ravel(),
                    p0=initial_guess)
                refined_z = z + popt[1]
                refined_y = y + popt[2]
                refined_x = x + popt[3]
                refined_peaks.append((refined_z, refined_y, refined_x))
            except Exception as e:
                refined_peaks.append((z, y, x))

        return refined_peaks

    def apply_gaussian_blur(self, input_float, sigma, shape):
        """
        Applies a Gaussian blur to the input image across multiple dimensions.
        Each dimension is blurred sequentially according to the provided sigma values.
        """
        blurred_image = input_float
        
        for i in range(shape):
            # Apply Gaussian filter with reflection at the borders for each dimension
            blurred_image = gaussian_filter(blurred_image, sigma=sigma[i], mode='reflect')
        
        return blurred_image
    
    def compute_sigma(self, steps, k, initial_sigma):
        """
        Computes a series of sigma values for Gaussian blurring.
        Each subsequent sigma is derived by multiplying the previous one by the factor k.
        """
        sigma = np.zeros(steps + 1)
        sigma[0] = initial_sigma

        for i in range(1, steps + 1):
            sigma[i] = sigma[i - 1] * k

        return sigma

    def compute_sigma_difference(self, sigma, image_sigma):
        """
        Computes the difference in sigma values required to achieve a desired level of blurring,
        accounting for the existing blur (image_sigma) in an image.
        """
        steps = len(sigma) - 1
        sigma_diff = np.zeros(steps + 1)
        sigma_diff[0] = np.sqrt(sigma[0]**2 - image_sigma**2)

        for i in range(1, steps + 1):
            sigma_diff[i] = np.sqrt(sigma[i]**2 - image_sigma**2)

        return sigma_diff
    
    def compute_sigmas(self, initial_sigma, shape):
        """
        Generates sigma values for Gaussian blurring across specified dimensions.
        Calculates the sigma differences required for sequential filtering steps.
        """
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
        """
        Normalizes an image to the [0, 1] range using predefined minimum and maximum intensities.
        """
        normalized_image = (image - self.min_intensity) / (self.max_intensity - self.min_intensity)
        return normalized_image

    def compute_difference_of_gaussian(self, image):
        """
        Computes feature points in an image using the Difference of Gaussian (DoG) method.
        """
        shape = 3
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

        # find peaks
        peaks = peak_local_max(dog, threshold_rel=min_initial_peak_value)
        
        # compare peaks with image data 
        # final_peaks = self.refine_peaks(peaks, image)
         
        return peaks 
    
    def interpolation(self, image, interest_points):
        """
        Interpolates the image at given interest points to retrieve their intensity values.
        Uses linear interpolation (order=1) and retrieves values at the nearest border if out-of-bounds.
        """
        if interest_points is None or len(interest_points) == 0:
            return np.array([]) 

        points = np.array(interest_points).T
        intensities = map_coordinates(image, points, order=1, mode='nearest')
        return intensities
    
    def upsample_coordinates(self, points, dsxy, dsz):
        """
        Upscales a list of coordinate points by specified scale factors for x, y (dsxy) and z (dsz) axes.
        """
        if points is None:
            return []
        return [(point[2] * dsxy, point[1] * dsxy, point[0] * dsz) for point in points]
    
    def run(self, image_chunk, dsxy, dsz):
        """
        Executes the entry point of the script.
        """
        final_peaks = self.compute_difference_of_gaussian(image_chunk)
        intensities = self.interpolation(image_chunk, final_peaks)
        upsampled_final_peaks = self.upsample_coordinates(final_peaks, dsxy, dsz)

        return {
            'interest_points': upsampled_final_peaks,
            'intensities': intensities
        }