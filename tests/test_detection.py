import unittest
from Rhapso.detection.difference_of_gaussian import main

class TestDetection(unittest.TestCase):
    def test_main(self):
        # Simulate arguments
        class Args:
            medianFilter = 10
            sigma = 1.8
            threshold = 0.05

        args = Args()
        main(args)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
