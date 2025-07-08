import unittest
from unittest.mock import patch, mock_open
from Rhapso.accuracy_metrics.threshold import Threshold

class TestThreshold(unittest.TestCase):

    def setUp(self):
        self.sample_data = {
            "alignment errors": {"minimum error": 5, "maximum error": 15, "mean error": 10},
            "Total IPS": 100,
            "Descriptive stats": {"Number of matches": 50},
            "KDE": {"minimum KDE": 0.1, "maximum KDE": 0.9},
            "Voxelization stats": {"Coefficient of Variation (CV)": 0.5}
        }

        self.threshold = Threshold(
            minimum_points=1, maximum_points=200,
            minimum_total_matches=10, maximum_total_matches=100,
            max_kde=1.0, min_kde=0.0,
            max_cv=1.0, min_cv=0.0,
            metric_path="dummy_path.json"
        )
        self.threshold.data = self.sample_data

    # @patch("sys.exit")
    # def test_check_alignment_within_range(self, mock_exit):
    #     self.threshold.check_alignment()
    #     mock_exit.assert_not_called()

    # @patch("sys.exit")
    # def test_check_alignment_out_of_range(self, mock_exit):
    #     self.threshold.data["alignment errors"]["min_error"] = -1
    #     self.threshold.check_alignment()
    #     mock_exit.assert_called_once_with(1)

    # @patch("sys.exit")
    # def test_check_alignment_no_min(self, mock_exit):
    #     threshold = Threshold(
    #         min_alignment=None, max_alignment=20,
    #         minimum_points=1, maximum_points=200,
    #         minimum_total_matches=10, maximum_total_matches=100,
    #         max_kde=1.0, min_kde=0.0,
    #         max_cv=1.0, min_cv=0.0,
    #         metric_path="dummy_path.json"
    #     )

    #     # Provide mock data
    #     threshold.data = {
    #         "alignment errors": {
    #             "min_error": 5,
    #             "max_error": 15,
    #             "mean_error": 10
    #         }
    #     }

    #     threshold.check_alignment()
    #     mock_exit.assert_not_called()
    
    # @patch("sys.exit")
    # def test_check_alignment_no_range(self, mock_exit):
    #     threshold = Threshold(
    #         min_alignment=None, max_alignment=None,
    #         minimum_points=1, maximum_points=200,
    #         minimum_total_matches=10, maximum_total_matches=100,
    #         max_kde=1.0, min_kde=0.0,
    #         max_cv=1.0, min_cv=0.0,
    #         metric_path="dummy_path.json"
    #     )

    #     # Provide mock data
    #     threshold.data = {
    #         "alignment errors": {
    #             "min_error": 5,
    #             "max_error": 15,
    #             "mean_error": 10
    #         }
    #     }

    #     threshold.check_alignment()
    #     mock_exit.assert_not_called()

    @patch("sys.exit")
    def test_check_points_within_range(self, mock_exit):
        self.threshold.check_points()
        mock_exit.assert_not_called()

    # @patch("sys.exit")
    # def test_check_points_no_range(self, mock_exit):
    #     threshold = Threshold(
    #         minimum_points=1, maximum_points=200,
    #         minimum_total_matches=None, maximum_total_matches=None,
    #         max_kde=1.0, min_kde=0.0,
    #         max_cv=1.0, min_cv=0.0,
    #         metric_path="dummy_path.json"
    #     )

    #     # Provide mock data
    #     threshold.data = {
    #         "alignment errors": {
    #             "minimum error": 5,
    #             "maximum error": 15,
    #             "mean error": 10
    #         }
    #     }

    #     threshold.check_alignment()
    #     mock_exit.assert_not_called()

    @patch("sys.exit")
    def test_check_points_out_of_range(self, mock_exit):
        self.threshold.data["Total IPS"] = 300
        self.threshold.check_points()
        mock_exit.assert_called_once_with(1)

    @patch("sys.exit")
    def test_check_matches_within_range(self, mock_exit):
        self.threshold.check_matches()
        mock_exit.assert_not_called()

    @patch("sys.exit")
    def test_check_matches_out_of_range(self, mock_exit):
        self.threshold.data["Descriptive stats"]["Number of matches"] = 0
        self.threshold.check_matches()
        mock_exit.assert_called_once_with(1)

    @patch("sys.exit")
    def test_check_kde_within_range(self, mock_exit):
        self.threshold.check_kde()
        mock_exit.assert_not_called()

    @patch("sys.exit")
    def test_check_kde_out_of_range(self, mock_exit):
        self.threshold.data["KDE"]["max"] = 2.0
        self.threshold.check_kde()
        mock_exit.assert_called_once_with(1)

    @patch("sys.exit")
    def test_check_cv_within_range(self, mock_exit):
        self.threshold.check_cv()
        mock_exit.assert_not_called()

    @patch("sys.exit")
    def test_check_cv_out_of_range(self, mock_exit):
        self.threshold.data["Voxelization stats"]["Coefficient of Variation (CV)"] = 2.0
        self.threshold.check_cv()
        mock_exit.assert_called_once_with(1)

    @patch("builtins.print")
    @patch("os.path.exists", return_value=False)
    def test_get_metric_json_file_not_found(self, mock_exists, mock_print):
        threshold = Threshold(
            minimum_points=1, maximum_points=200,
            minimum_total_matches=10, maximum_total_matches=100,
            max_kde=1.0, min_kde=0.0,
            max_cv=1.0, min_cv=0.0,
            metric_path="nonexistent.json"
        )

        threshold.get_metric_json()

        mock_print.assert_called_once_with("File not found: nonexistent.json")

if __name__ == "__main__":
    unittest.main()
