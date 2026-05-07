import unittest
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates
from memory_profiler import profile


from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian


class TestDifferenceOfGaussian(unittest.TestCase):
    def setUp(self):
        self.dog = DifferenceOfGaussian(
            min_intensity=0, max_intensity=255, sigma=1.0, threshold=0.5,
            median_filter=False, mip_map_downsample=np.eye(4),
        )
        self.image = np.random.rand(10, 10, 10) * 255

    def test_normalize_image(self):
        normalized_image = self.dog.normalize_image(self.image)
        self.assertTrue(np.all(normalized_image >= 0) and np.all(normalized_image <= 1))

    def test_compute_sigma(self):
        sigma = self.dog.compute_sigma(3, 2, 1.0)
        expected_sigma = np.array([1.0, 2.0, 4.0, 8.0])
        np.testing.assert_array_almost_equal(sigma, expected_sigma)

    def test_compute_sigma_difference(self):
        sigma = np.array([1.0, 2.0, 4.0, 8.0])
        sigma_diff = self.dog.compute_sigma_difference(sigma, 0.5)
        expected_sigma_diff = np.sqrt(sigma**2 - 0.5**2)
        np.testing.assert_array_almost_equal(sigma_diff, expected_sigma_diff)

    def test_apply_gaussian_blur(self):
        sigma = np.array([1.0, 1.0, 1.0])
        blurred_image = self.dog.apply_gaussian_blur(self.image, sigma, 3)
        self.assertEqual(blurred_image.shape, self.image.shape)

    def test_compute_difference_of_gaussian(self):
        peaks = self.dog.compute_difference_of_gaussian(self.image)
        self.assertIsInstance(peaks, np.ndarray)

    def test_interpolation(self):
        points = np.array(
            [
                [468.345027587921, 488.36223951244153, 2.970786928172377],
                [856.6701082186948, 416.01488311517676, 3.4227515981883694],
            ]
        )
        intensities = self.dog.interpolation(self.image, points)
        self.assertEqual(len(intensities), len(points))

    def test_upsample_coordinates(self):
        points = np.array(
            [
                [468.345027587921, 488.36223951244153, 2.970786928172377],
                [856.6701082186948, 416.01488311517676, 3.4227515981883694],
            ]
        )
        upsampled_points = self.dog.upsample_coordinates(points)
        self.assertEqual(len(upsampled_points), len(points))


    def test_lower_bounds_added_after_upsample(self):
        """Regression: apply_lower_bounds must happen AFTER
        upsample_coordinates so that lb (L0 coords) is added to
        L0-upsampled peaks, not to downsampled-space peaks that then
        get multiplied by the pyramid scale factor.

        Prior bug: lb was added before upsample, inflating the lb
        component by the scale factor and producing Z-banded IPs
        when chunks_per_bound > 1.
        """
        scale = 16
        half_pixel = 0.5 * (scale - 1)
        mip_map_downsample = np.array([
            [scale, 0, 0, half_pixel],
            [0, scale, 0, half_pixel],
            [0, 0, scale, half_pixel],
            [0, 0, 0, 1],
        ], dtype=float)

        dog = DifferenceOfGaussian(
            min_intensity=0, max_intensity=255, sigma=1.0,
            threshold=0.5, median_filter=False,
            mip_map_downsample=mip_map_downsample,
        )

        # Synthetic peak at downsampled coord (5, 3, 2)
        peaks = np.array([[5.0, 3.0, 2.0]], dtype=np.float32)
        lb = [0, 0, 99]  # L0 coords — chunk starts at Z=99
        offset = 0

        # Correct transform order: upsample then add lb
        result = dog.upsample_coordinates(peaks)
        result = dog.apply_lower_bounds(result, lb)
        result = dog.apply_offset(result, offset)

        expected_x = 5.0 * scale + half_pixel + 0
        expected_y = 3.0 * scale + half_pixel + 0
        expected_z = 2.0 * scale + half_pixel + 99

        np.testing.assert_allclose(
            result[0],
            [expected_x, expected_y, expected_z],
            atol=1e-4,
            err_msg="lb must be added AFTER upsample to avoid inflating "
                    "the offset by the pyramid scale factor",
        )

        # Verify the OLD (buggy) order would give a wrong answer
        buggy = dog.apply_lower_bounds(peaks.copy(), lb)
        buggy = dog.apply_offset(buggy, offset)
        buggy = dog.upsample_coordinates(buggy)
        buggy_z = float(buggy[0, 2])
        correct_z = float(result[0, 2])
        # Buggy Z = (2 + 99) * 16 + 7.5 = 1623.5 vs correct 138.5
        self.assertNotAlmostEqual(
            buggy_z, correct_z, places=1,
            msg="Old transform order must differ from correct order "
                "when lb has non-zero Z",
        )


if __name__ == "__main__":
    unittest.main()
