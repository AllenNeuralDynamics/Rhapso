import unittest
from unittest.mock import patch

from Rhapso.accuracy_metrics.alignment_threshold import AlignmentThreshold


class TestThreshold(unittest.TestCase):

    def setUp(self):
        self.sample_data = {
            "alignment errors": {"minimum error": 5, "maximum error": 15, "mean error": 10},
            "Total IPS": 100,
            "Descriptive stats": {"Number of matches": 50},
            "KDE": {"minimum KDE": 0.1, "maximum KDE": 0.9},
            "Voxelization stats": {"Coefficient of Variation (CV)": 0.5}
        }

        self.threshold = AlignmentThreshold(
            min_alignment=0, max_alignment=20,
            metric_path="dummy_path.json"
        )
        self.threshold.data = self.sample_data

    @patch("sys.exit")
    def test_check_alignment_within_range(self, mock_exit):
        self.threshold.check_alignment()
        mock_exit.assert_not_called()

    @patch("sys.exit")
    def test_check_alignment_out_of_range(self, mock_exit):
        self.threshold.data["alignment errors"]["minimum error"] = -1
        self.threshold.check_alignment()
        mock_exit.assert_called_once_with(1)

    @patch("sys.exit")
    def test_check_alignment_no_min(self, mock_exit):
        threshold = AlignmentThreshold(
            min_alignment=None, max_alignment=20,
            metric_path="dummy_path.json"
        )

        # Provide mock data
        threshold.data = {
            "alignment errors": {
                "minimum error": 5,
                "maximum error": 15,
                "mean_error": 10
            }
        }

        threshold.check_alignment()
        mock_exit.assert_not_called()
    
    @patch("sys.exit")
    def test_check_alignment_no_range(self, mock_exit):
        threshold = AlignmentThreshold(
            min_alignment=None, max_alignment=None,
            metric_path="dummy_path.json"
        )

        # Provide mock data
        threshold.data = {
            "alignment errors": {
                "minimum error": 5,
                "maximum error": 15,
            }
        }

        threshold.check_alignment()
        mock_exit.assert_not_called()

if __name__ == "__main__":
    unittest.main()
