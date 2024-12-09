import unittest
from Rhapso.PhaseCorrelation import phase_correlation

class TestPhaseCorrelation(unittest.TestCase):

    def test_phase_correlation_identical_images(self):
        image1 = "image_data_1"
        image2 = "image_data_1"
        result = phase_correlation(image1, image2)
        self.assertTrue(result)

    def test_phase_correlation_different_images(self):
        image1 = "image_data_1"
        image2 = "image_data_2"
        result = phase_correlation(image1, image2)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()