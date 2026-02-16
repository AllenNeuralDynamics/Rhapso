from scipy.ndimage import gaussian_filter 
from scipy.ndimage import map_coordinates
from scipy.ndimage import median_filter
from scipy.ndimage import maximum_filter
from scipy.linalg import lu_factor, lu_solve
import numpy as np

"""
Difference of Gaussian computes the difference of gaussian on a 3D image chunk, collecting interest points and intensities
"""

class DifferenceOfGaussian:
    def __init__(self, min_intensity, max_intensity, sigma, threshold, median_filter, mip_map_downsample):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.sigma = sigma
        self.threshold = threshold
        self.median_filter = median_filter
        self.mip_map_downsample = mip_map_downsample
    
    def apply_offset(self, peaks, offset_z):
        """
        Updates points with sub-regional offset
        """
        if peaks is None or peaks.size == 0:
            return peaks

        peaks = np.asarray(peaks, dtype=np.float32).copy()
        peaks[:, 2] += offset_z

        return peaks

    def upsample_coordinates(self, points):
        """
        Map 3D points from downsampled (mipmap) space back to full-res
        """
        P = np.asarray(points, dtype=np.float32)        
        T = np.asarray(self.mip_map_downsample, dtype=np.float32)

        R = T[:3, :3]
        t = T[:3, 3]

        return (P @ R.T) + t
    
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

    def quadratic_fit(self, image, position):
        """
        Compute the gradient vector (g) and Hessian matrix (H) at an integer voxel using second-order central
        """
        n = len(position)
        g = np.zeros(n, dtype=np.float64)
        H = np.zeros((n, n), dtype=np.float64)

        a1 = float(image[tuple(position)])  # center value

        for d in range(n):
            pos = list(position)
            pos[d] -= 1
            a0 = float(image[tuple(pos)])
            pos[d] += 2
            a2 = float(image[tuple(pos)])

            # g(d) = (a2 - a0)/2
            g[d] = 0.5 * (a2 - a0)

            # H(dd) = a2 - 2*a1 + a0
            H[d, d] = a2 - 2.0 * a1 + a0

            # Off-diagonals: ( +1,+1 ), ( -1,+1 ), ( +1,-1 ), ( -1,-1 )
            for e in range(d + 1, n):
                vals = []
                for off_d, off_e in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                    pos = list(position)
                    pos[d] += off_d
                    pos[e] += off_e
                    vals.append(float(image[tuple(pos)]))
                v = (vals[0] - vals[1] - vals[2] + vals[3]) * 0.25
                H[d, e] = H[e, d] = v

        return g, H, a1

    def refine_peaks(self, peaks, image):
        """
        Quadratic peak refinement - iteratively refine integer-voxel peaks to subpixel locations using a 
        quadratic (Newton) update from the local gradient/Hessian, with step capping and boundary checks
        """
        max_moves=10
        maxima_tolerance=0.1
        threshold=0.0        
        return_invalid_peaks=False
        allowed_to_move_in_dim=None

        if allowed_to_move_in_dim is None:
            allowed_to_move_in_dim = [True] * image.ndim

        refined_positions = []
        shape = np.asarray(image.shape, dtype=np.int64)
        n = image.ndim

        def solve_Hg(H, g):
            lu, piv = lu_factor(H)
            return -lu_solve((lu, piv), g)

        for peak in peaks:
            base = getattr(peak, "location", peak)       
            position = np.array(base, dtype=np.int64)    
            stable = False

            for move in range(max_moves):
                # need interior neighborhood (±1 available)
                if np.any(position < 1) or np.any(position >= shape - 1):
                    break

                g, H, a1 = self.quadratic_fit(image, position)
                offset = solve_Hg(H, g)

                threshold_move = 0.5 + move * float(maxima_tolerance)

                stable = True
                for d in range(n):
                    if allowed_to_move_in_dim[d] and abs(offset[d]) > threshold_move:
                        position[d] += 1 if offset[d] > 0.0 else -1
                        stable = False

                if stable:
                    # value at subpixel = center + 0.5 * g^T * offset
                    value = float(a1 + 0.5 * np.dot(g, offset))
                    if abs(value) > float(threshold):
                        refined_positions.append(position.astype(np.float64) + offset)
                    # whether kept or filtered by threshold, we’re done with this peak
                    break

            if (not stable) and return_invalid_peaks:
                # invalid handling: return original integer location
                refined_positions.append(np.asarray(base, dtype=np.float64))

        if not refined_positions:
            return np.empty((0, n), dtype=np.float32)

        return np.vstack(refined_positions).astype(np.float32)

    def find_peaks(self, dog, min_initial_peak_value):
        """
        Find 3D peak candidates as strict local maxima (26-neighborhood)
        """
        L = np.asarray(dog, dtype=np.float32)

        # skip outer 1-voxel border 
        interior = np.zeros_like(L, dtype=bool)
        interior[1:-1, 1:-1, 1:-1] = True

        # strict local maxima vs 26 neighbors (exclude center)
        fp = np.ones((3, 3, 3), dtype=bool)
        fp[1, 1, 1] = False
        neigh_max = maximum_filter(L, footprint=fp, mode="reflect")
        strict_max = L > neigh_max

        strong = L >= float(min_initial_peak_value)
        cand = interior & strict_max & strong
        
        if not cand.any():
            return np.empty((0, 3), dtype=np.int32)

        peaks = np.column_stack(np.nonzero(cand)).astype(np.int32)
        
        return peaks

    def apply_gaussian_blur(self, img, sigma):
        """
        Apply an N-D Gaussian blur with per-axis sigmas using reflect padding at the borders
        """
        sigma = tuple(float(s) for s in sigma)
        blurred_image = gaussian_filter(img, sigma=sigma, mode='reflect')
        
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
        min_initial_peak_value = np.float32(self.threshold) / np.float32(3.0)
        k = 2 ** (1 / 4)
        k_min_1_inv = 1.0 / (k - 1.0)

        # normalize image using min/max intensities
        input_float = self.normalize_image(image)                                            

        # calculate gaussian blur levels 
        sigma_1, sigma_2 = self.compute_sigmas(self.sigma, shape, k)                    

        # apply gaussian blur
        blurred_image_1 = self.apply_gaussian_blur(input_float, sigma_1)                
        blurred_image_2 = self.apply_gaussian_blur(input_float, sigma_2)

        # subtract blurred images
        dog = (blurred_image_1 - blurred_image_2) * k_min_1_inv

        # get all peaks
        peaks = self.find_peaks(dog, min_initial_peak_value)

        # localize peaks
        final_peak_values = self.refine_peaks(peaks, dog)
         
        return final_peak_values

    def background_subtract_xy(self, image_chunk):
        """
        Remove slow-varying background in XY by subtracting a medianfilter
        """
        r = int(self.median_filter or 0)
        img = image_chunk.astype(np.float32, copy=False)
        if r <= 0:
            return img

        k = 2 * r + 1

        # 1) Add XY border (reflect), no padding in Z
        pad = ((r, r), (r, r), (0, 0))
        img_pad = np.pad(img, pad, mode='reflect')

        # 2) Median background on padded image (XY-only)
        bg = median_filter(img_pad, size=(k, k, 1), mode='reflect')

        # 3) Subtract and crop back to original core
        sub = img_pad - bg
        
        return sub[r:-r, r:-r, :]

    def run(self, image_chunk, offset, lb):
        """
        Executes the entry point of the script.
        """
        image_chunk = self.background_subtract_xy(image_chunk)
        peaks = self.compute_difference_of_gaussian(image_chunk)

        if peaks.size == 0:
            intensities = np.empty((0,), dtype=image_chunk.dtype)
            final_peaks = peaks

        else:
            intensities = map_coordinates(image_chunk, peaks.T, order=1, mode='reflect')
            final_peaks = self.apply_lower_bounds(peaks, lb)
            final_peaks = self.apply_offset(final_peaks, offset)
            final_peaks = self.upsample_coordinates(final_peaks)

        return {
            'interest_points': final_peaks,
            'intensities': intensities
        }