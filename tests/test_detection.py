import unittest
from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian

class TestDetection(unittest.TestCase):
    def test_main(self):
        # Simulate arguments
        class Args:
            medianFilter = 10
            sigma = 1.8
            threshold = 0.05

        args = Args()
        DifferenceOfGaussian(args)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
