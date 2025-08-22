from scipy.ndimage import gaussian_filter 
from skimage.feature import peak_local_max
from scipy.ndimage import map_coordinates
from scipy.ndimage import median_filter
import numpy as np

"""
Utility class to compute difference of gaussian on a 3D image chunk, collecting interest points and intensities
"""

class DifferenceOfGaussian:
    def __init__(self, min_intensity, max_intensity, sigma, threshold, median_filter):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.sigma = sigma
        self.threshold = threshold
        self.median_filter = median_filter
    
    def apply_offset(self, peaks, offset_z):
        """
        Updates points with sub-regional offset
        """
        if peaks is None or peaks.size == 0:
            return peaks

        peaks = np.asarray(peaks, dtype=np.float32).copy()
        peaks[:, 0] += offset_z

        return peaks
    
    def upsample_coordinates(self, points, dsxy, dsz):
        """
        Upsamples points back to original space
        """
        if points is None or len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        points = np.asarray(points, dtype=np.float32)
        
        def calc_shift(factor):
            return 0.5 * (factor.bit_length() - 1) if factor > 1 else 0

        shift_z = calc_shift(dsz)
        shift_y = calc_shift(dsxy)
        shift_x = calc_shift(dsxy)

        upsampled = np.empty_like(points)
        upsampled[:, 2] = points[:, 2] * dsz + shift_z  
        upsampled[:, 1] = points[:, 1] * dsxy + shift_y  
        upsampled[:, 0] = points[:, 0] * dsxy + shift_x  

        return upsampled
    
    def apply_lower_bounds(self, peaks, lower_bounds):
        """
        Updates points with lower bounds 
        """
        if peaks is None or peaks.size == 0:
            return peaks

        peaks = np.asarray(peaks, dtype=np.float32).copy()
        bounds_xyz = np.array(lower_bounds, dtype=np.float32)
        peaks += bounds_xyz

        return peaks
    
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
        Refines the position of detected peaks using quadratic localization.
        """
        if len(peaks) == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        max_moves= 10
        tolerance= 0.01

        # peaks = np.asarray([p["coords"] for p in peaks], dtype=int)
        peaks = peaks.astype(np.int64, copy=False)

        padded = np.pad(image, 1, mode='reflect')
        refined = []
        
        for peak in peaks:
            current = np.array(peak, dtype=int) + 1  # account for padding
            converged = False
            
            for move in range(max_moves):
                # Check bounds
                if np.any(current < 1) or np.any(current >= np.array(padded.shape) - 1):
                    break
                
                # Get center value
                center_val = padded[tuple(current)]
                
                # Compute gradient and Hessian
                grad = np.zeros(3)
                H = np.zeros((3, 3))
                
                # For each dimension
                for d in range(3):
                    # Create index arrays for accessing neighbors
                    idx_minus = list(current)
                    idx_plus = list(current)
                    idx_minus[d] -= 1
                    idx_plus[d] += 1
                    
                    # Gradient: g(d) = (a2 - a0) / 2
                    a0 = padded[tuple(idx_minus)]
                    a2 = padded[tuple(idx_plus)]
                    grad[d] = (a2 - a0) * 0.5
                    
                    # Diagonal Hessian: H(d,d) = a2 - 2*a1 + a0
                    H[d, d] = a2 - 2 * center_val + a0
                    
                    # Off-diagonal elements
                    for e in range(d + 1, 3):
                        # Create indices for 2D cross pattern
                        idx_pp = list(current)
                        idx_pp[d] += 1
                        idx_pp[e] += 1
                        
                        idx_pm = list(current)
                        idx_pm[d] += 1
                        idx_pm[e] -= 1
                        
                        idx_mp = list(current)
                        idx_mp[d] -= 1
                        idx_mp[e] += 1
                        
                        idx_mm = list(current)
                        idx_mm[d] -= 1
                        idx_mm[e] -= 1
                        
                        # H(d,e) = (a2b2 - a0b2 - a2b0 + a0b0) / 4
                        val = (padded[tuple(idx_pp)] - padded[tuple(idx_pm)] - 
                            padded[tuple(idx_mp)] + padded[tuple(idx_mm)]) * 0.25
                        H[d, e] = val
                        H[e, d] = val
                
                # Solve for offset
                try:
                    offset = -np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    offset = np.zeros(3)
                
                # Check convergence 
                converged = True
                threshold_move = 0.5 + move * tolerance
                
                for d in range(3):
                    if abs(offset[d]) > threshold_move:
                        # Move by integer step in the direction of offset
                        current[d] += 1 if offset[d] > 0 else -1
                        converged = False
                
                if converged:
                    break
            
            if converged:
                # Calculate value using quadratic approximation
                # value = f(x0) + 0.5 * g^T * offset
                value = center_val + 0.5 * np.dot(grad, offset)
                
                if abs(value) > self.threshold:
                    # Return position as float with subpixel offset
                    refined_pos = current.astype(float) + offset - 1  # subtract padding
                    refined.append(refined_pos)
        
        # return = np.array(refined, dtype=np.float32)
        refined = np.asarray(refined, dtype=np.float32).reshape(-1, 3)
        return refined
    
    def find_peaks(self, dog, min_initial_peak_value):
        """
        Detects local maxima in a 3D DoG array above a threshold, filtering for positive values.
        """
        footprint = np.ones((3, 3, 3), dtype=bool)
        peaks = peak_local_max(
            dog,
            min_distance=1,
            threshold_abs=min_initial_peak_value,
            footprint=footprint,
            exclude_border=False,
        )
    
        if peaks.size == 0:
            return np.empty((0, 3), dtype=np.int32)  

        # Filter peaks with positive value
        values = dog[tuple(peaks.T)]
        valid_mask = values > 0
        return peaks[valid_mask].astype(np.int32)

    def apply_gaussian_blur(self, input_float, sigma, shape):
        """
        Applies a Gaussian blur to the input image across multiple dimensions.
        Each dimension is blurred sequentially according to the provided sigma values.
        """
        blurred_image = input_float
        
        for i in range(shape):
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
    
    def compute_sigmas(self, initial_sigma, shape, k):
        """
        Generates sigma values for Gaussian blurring across specified dimensions.
        Calculates the sigma differences required for sequential filtering steps.
        """
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
        shape = len(image.shape)
        min_initial_peak_value = self.threshold / 3
        k = 2 ** (1 / 4)
        k_min_1_inv = 1.0 / (k - 1.0)

        # normalize image using min/max intensities
        input_float = self.normalize_image(image)                                            

        # calculate gaussian blur levels 
        sigma_1, sigma_2 = self.compute_sigmas(self.sigma, shape, k)                    

        # apply gaussian blur
        blurred_image_1 = self.apply_gaussian_blur(input_float, sigma_1, shape)                
        blurred_image_2 = self.apply_gaussian_blur(input_float, sigma_2, shape)

        # subtract blurred images
        dog = (blurred_image_2 - blurred_image_1) * k_min_1_inv

        # get all peaks
        peaks = self.find_peaks(dog, min_initial_peak_value)

        # localize peaks
        final_peak_values = self.refine_peaks(peaks, dog)
         
        return final_peak_values
    
    def background_subtract_xy(self, image_chunk):
        img = image_chunk.astype(np.float32, copy=False)

        # Median only in XY
        k = 2 * self.median_filter + 1
        bg = median_filter(img, size=(1, k, k), mode='reflect')
        out = img - bg
        
        return out

    def run(self, image_chunk, dsxy, dsz, offset, lb):
        """
        Executes the entry point of the script.
        """
        # image_chunk = self.background_subtract_xy(image_chunk)
        peaks = self.compute_difference_of_gaussian(image_chunk)

        if peaks.size == 0:
            intensities = np.empty((0,), dtype=image_chunk.dtype)
            final_peaks = peaks

        else:
            intensities = map_coordinates(image_chunk, peaks.T, order=1, mode='reflect')
            final_peaks = self.apply_lower_bounds(peaks, lb)
            final_peaks = self.upsample_coordinates(final_peaks, dsxy, dsz)
            final_peaks = self.apply_offset(final_peaks, offset)

        return {
            'interest_points': final_peaks,
            'intensities': intensities
        }