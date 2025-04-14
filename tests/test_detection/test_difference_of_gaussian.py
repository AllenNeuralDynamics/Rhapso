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
            min_intensity=0, max_intensity=255, sigma=1.0, threshold=0.5
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
        upsampled_points = self.dog.upsample_coordinates(points, 2, 1)
        self.assertEqual(len(upsampled_points), len(points))


if __name__ == "__main__":
    unittest.main()
