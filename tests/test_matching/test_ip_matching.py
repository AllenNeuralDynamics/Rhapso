import logging
import unittest
from unittest.mock import MagicMock, patch
from xml.etree.ElementTree import Element, SubElement
from unittest.mock import patch, MagicMock
import numpy as np
import xml.etree.ElementTree as ET


from Rhapso.matching.interest_point_matching import (
    buildLabelMap,
    compute_matches,
    group_views,
    list_files_under_prefix,
    open_n5_dataset,
    parse_and_read_datasets,
    perform_pairwise_matching,
    print_dataset_info,
    ransac_filter_matches,
    start_matching,
)


class TestInterestPointMatching(unittest.TestCase):

    @patch("builtins.print")
    def test_list_files_under_prefix_key_error(self, mock_print):
        node = {}
        list_files_under_prefix(node, "/")
        mock_print.assert_any_call("No items found under the path /")

    @patch("builtins.print")
    def test_no_view_setup_elements(self, mock_print):
        root = Element("root")

        timepoint1 = SubElement(root, "Timepoint")
        SubElement(timepoint1, "id").text = "0"

        label_map, label_weights = buildLabelMap(root)

        self.assertEqual(label_map, {})
        self.assertEqual(label_weights, {})
        mock_print.assert_any_call("No ViewSetup elements found in the XML.")

    @patch("builtins.print")
    def test_no_timepoint_elements(self, mock_print):
        root = Element("root")

        view_setup1 = SubElement(root, "ViewSetup")
        SubElement(view_setup1, "id").text = "1"
        SubElement(view_setup1, "name").text = "label1"

        label_map, label_weights = buildLabelMap(root)

        self.assertEqual(label_map, {})
        self.assertEqual(label_weights, {})
        mock_print.assert_any_call("No Timepoint elements found in the XML.")

    @patch("builtins.print")
    def test_build_label_map(self, mock_print):
        # Create a mock XML structure
        root = Element("root")

        view_setup1 = SubElement(root, "ViewSetup")
        SubElement(view_setup1, "id").text = "1"
        SubElement(view_setup1, "name").text = "label1"

        timepoint1 = SubElement(root, "Timepoint")
        SubElement(timepoint1, "id").text = "0"

        label_map, label_weights = buildLabelMap(root)

        expected_label_map = {(0, 1): {"label1": 1.0}}
        expected_label_weights = {"label1": 1.0}

        self.assertEqual(label_map, expected_label_map)
        self.assertEqual(label_weights, expected_label_weights)
        mock_print.assert_any_call("building label map")

    def test_group_views_normal_case(self):
        views = [(18, 1), (18, 2), (30, 1), (30, 3)]
        expected_output = {18: [1, 2], 30: [1, 3]}

        self.assertEqual(group_views(views), expected_output)

    def test_group_views_empty_list(self):
        views = []
        expected_output = {}
        self.assertEqual(group_views(views), expected_output)

    def test_group_views_single_view(self):
        views = [(0, 1)]
        expected_output = {0: [1]}
        self.assertEqual(group_views(views), expected_output)

    def test_group_views_multiple_views_same_timepoint(self):
        views = [(0, 1), (0, 1), (0, 2)]
        expected_output = {0: [1, 1, 2]}
        self.assertEqual(group_views(views), expected_output)

    @patch("builtins.print")
    @patch("zarr.open")
    @patch("zarr.N5Store")
    def test_open_n5_dataset_local(self, mock_n5store, mock_zarr_open, mock_print):
        mock_store = MagicMock()
        mock_n5store.return_value = mock_store
        mock_dataset = MagicMock()
        mock_zarr_open.return_value = mock_dataset

        result = open_n5_dataset("/path/to/local/n5")
        mock_n5store.assert_called_once_with("/path/to/local/n5")
        mock_zarr_open.assert_called_once_with(mock_store, mode="r")
        mock_print.assert_called_once_with(
            "✅ Successfully opened N5 dataset at /path/to/local/n5"
        )
        self.assertEqual(result, mock_dataset)

    @patch("builtins.print")
    @patch("s3fs.S3Map")
    @patch("s3fs.S3FileSystem")
    @patch("zarr.open")
    def test_open_n5_dataset_s3(
        self, mock_zarr_open, mock_s3fs, mock_s3map, mock_print
    ):
        mock_s3 = MagicMock()
        mock_s3fs.return_value = mock_s3
        mock_store = MagicMock()
        mock_s3map.return_value = mock_store
        mock_dataset = MagicMock()
        mock_zarr_open.return_value = mock_dataset

        result = open_n5_dataset("s3://bucket/path/to/n5")
        mock_s3fs.assert_called_once_with(anon=False)
        mock_s3map.assert_called_once_with(
            root="s3://bucket/path/to/n5", s3=mock_s3, check=False
        )
        mock_zarr_open.assert_called_once_with(mock_store, mode="r")
        mock_print.assert_called_once_with(
            "✅ Successfully opened N5 dataset at s3://bucket/path/to/n5"
        )
        self.assertEqual(result, mock_dataset)

    @patch("builtins.print")
    def test_open_n5_dataset_exception(self, mock_print):
        with patch("zarr.open", side_effect=Exception("Test error")):
            result = open_n5_dataset("/path/to/local/n5")
            mock_print.assert_called_once_with(
                "❌ Error opening N5 dataset at /path/to/local/n5: Test error"
            )
            self.assertIsNone(result)

    @patch("builtins.print")
    def test_print_dataset_info_valid(self, mock_print):
        dataset = MagicMock()
        dataset.shape = (100, 200)
        dataset.domain = "example_domain"
        dataset.dtype = "float32"
        dataset.read().result.return_value = "mock_data"

        print_dataset_info(dataset, "test_label")

        mock_print.assert_any_call("\n📊 Dataset Info (test_label):")
        mock_print.assert_any_call("   Number of items: 100")
        mock_print.assert_any_call("   Shape: (100, 200)")
        mock_print.assert_any_call("   Dataset Domain: example_domain")
        mock_print.assert_any_call("   Dataset Properties:")
        mock_print.assert_any_call("     Data Type: float32")
        mock_print.assert_any_call("     Shape: (100, 200)")
        mock_print.assert_any_call("   🟢 Raw Data (NumPy Array):\n", "mock_data")

    @patch("builtins.print")
    def test_print_dataset_info_invalid(self, mock_print):
        dataset = "invalid_dataset"

        print_dataset_info(dataset, "test_label")

        mock_print.assert_any_call(
            "❌ Error retrieving dataset info: Invalid dataset object for label 'test_label'. Expected an object with a 'shape' attribute, got <class 'str'>."
        )

    @patch("builtins.print")
    def test_print_dataset_info_exception(self, mock_print):
        dataset = MagicMock()
        dataset.shape = (100, 200)
        dataset.read.side_effect = Exception("Test error")

        print_dataset_info(dataset, "test_label")

        mock_print.assert_any_call("❌ Error retrieving dataset info: Test error")

    @patch("builtins.print")
    @patch("zarr.open")
    @patch("zarr.N5Store")
    @patch("os.path.join", return_value="/mock/path/to/n5")
    @patch("Rhapso.matching.interest_point_matching.parse_xml")
    def test_parse_and_read_datasets_local(
        self,
        mock_parse_xml,
        mock_os_path_join,  # This is being used even if it looks like it is not!
        mock_n5store,
        mock_zarr_open,
        mock_print,
    ):
        mock_parse_xml.return_value = {(0, 1): {"path": "mock_path"}}
        mock_store = MagicMock()
        mock_n5store.return_value = mock_store
        mock_dataset = MagicMock()
        mock_zarr_open.return_value = {
            "mock_path/beads/interestpoints/loc": mock_dataset
        }
        mock_dataset.shape = (100, 200)

        interest_point_info, view_paths = parse_and_read_datasets(
            "mock.xml", "/path/to/local/n5"
        )

        expected_interest_point_info = {
            (0, 1): {
                "loc": {"num_items": 100, "shape": (100, 200), "data": mock_dataset}
            }
        }
        expected_view_paths = {(0, 1): {"path": "mock_path"}}

        self.assertEqual(interest_point_info, expected_interest_point_info)
        self.assertEqual(view_paths, expected_view_paths)
        mock_print.assert_any_call(
            "\n🔍 Found 1 view ID/timepoint interest point folders to analyze."
        )
        mock_print.assert_any_call("\n🔗 Processing view 1/1: Setup 1, Timepoint 0")
        mock_print.assert_any_call("🛠  Loading dataset from: /mock/path/to/n5")

    @patch("builtins.print")
    def test_compute_matches_valid(self, mock_print):
        pointsA = np.array([[1, 2], [3, 4], [5, 6]])
        pointsB = np.array([[1, 2], [3, 4], [7, 8]])
        expected_matches = [(0, 0), (1, 1)]

        matches = compute_matches(pointsA, pointsB)
        self.assertEqual(matches, expected_matches)
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_compute_matches_invalid_pointsA(self, mock_print):
        pointsA = None
        pointsB = np.array([[1, 2], [3, 4], [7, 8]])

        matches = compute_matches(pointsA, pointsB)
        self.assertEqual(matches, [])
        mock_print.assert_called_once_with("Invalid points data provided for matching.")

    @patch("builtins.print")
    def test_compute_matches_invalid_pointsB(self, mock_print):
        pointsA = np.array([[1, 2], [3, 4], [5, 6]])
        pointsB = None

        matches = compute_matches(pointsA, pointsB)
        self.assertEqual(matches, [])
        mock_print.assert_called_once_with("Invalid points data provided for matching.")

    def test_compute_matches_no_matches(self):
        pointsA = np.array([[1, 2], [3, 4], [5, 6]])
        pointsB = np.array([[10, 20], [30, 40], [50, 60]])

        matches = compute_matches(pointsA, pointsB)
        self.assertEqual(matches, [])

    @patch("builtins.print")
    def test_ransac_filter_matches_valid(self, mock_print):
        pointsA = np.array([[1, 2], [3, 4], [5, 6]])
        pointsB = np.array([[2, 3], [4, 5], [6, 7]])
        matches = [[0, 0], [1, 1], [2, 2]]

        filtered_matches, translation = ransac_filter_matches(pointsA, pointsB, matches)
        expected_translation = np.array([1, 1])

        self.assertEqual(filtered_matches, matches)
        np.testing.assert_array_equal(translation, expected_translation)
        mock_print.assert_any_call(
            "🔎 RANSAC estimated translation: [1. 1.] with 3 inliers out of 3 matches."
        )
        mock_print.assert_any_call(
            "✅ RANSAC filtering retained 3 matches after outlier removal."
        )

    @patch("builtins.print")
    def test_ransac_filter_matches_no_matches(self, mock_print):
        pointsA = np.array([[1, 2], [3, 4], [5, 6]])
        pointsB = np.array([[2, 3], [4, 5], [6, 7]])
        matches = []

        filtered_matches, translation = ransac_filter_matches(pointsA, pointsB, matches)

        self.assertEqual(filtered_matches, [])
        self.assertIsNone(translation)

    # Rhapso/matching/interest_point_matching.py
    @patch("builtins.print")
    @patch("Rhapso.matching.interest_point_matching.compute_matches")
    @patch("Rhapso.matching.interest_point_matching.ransac_filter_matches")
    def test_perform_pairwise_matching(
        self, mock_ransac_filter_matches, mock_compute_matches, mock_print
    ):
        interest_point_info = {
            (0, 1): {
                "loc": {
                    "data": np.array([[1, 2, 3], [4, 5, 6]]),
                    "num_items": 2,
                    "shape": (936, 3),
                }
            },
            (0, 2): {
                "loc": {
                    "data": np.array([[1, 2, 3], [4, 5, 6]]),
                    "num_items": 2,
                    "shape": (936, 3),
                }
            },
        }
        view_paths = {}
        all_matches = []
        labels = []
        method = "test_method"

        mock_compute_matches.return_value = [(0, 0), (1, 1)]
        mock_ransac_filter_matches.return_value = (
            [(0, 0), (1, 1)],
            np.array([1, 1, 1]),
        )

        perform_pairwise_matching(
            interest_point_info, view_paths, all_matches, labels, method
        )

        self.assertEqual(len(all_matches), 2)
        self.assertEqual(all_matches[0], ((0, 1), (0, 2), 0, 0))
        self.assertEqual(all_matches[1], ((0, 1), (0, 2), 1, 1))
        mock_print.assert_any_call("\n🔗 Starting pairwise matching across views:")
        mock_print.assert_any_call("\n⏱️  Processing timepoint 0 with 2 views")
        mock_print.assert_any_call("\n💥 Matching view 1 with view 2 at timepoint 0")
        mock_print.assert_any_call(
            "🔍 Computing initial matches using nearest neighbors..."
        )
        mock_print.assert_any_call("⚙️ Found 2 initial matches.")
        mock_print.assert_any_call(
            "Matched Points (Global Coordinates): [ViewSetupId: 1, TimePointId: 0, x: 1.00, y: 2.00, z: 3.00] <=> [ViewSetupId: 2, TimePointId: 0, x: 1.00, y: 2.00, z: 3.00]"
        )
        mock_print.assert_any_call(
            "Matched Points (Global Coordinates): [ViewSetupId: 1, TimePointId: 0, x: 4.00, y: 5.00, z: 6.00] <=> [ViewSetupId: 2, TimePointId: 0, x: 4.00, y: 5.00, z: 6.00]"
        )

    @patch("builtins.print")
    @patch("Rhapso.matching.interest_point_matching.compute_matches")
    @patch("Rhapso.matching.interest_point_matching.ransac_filter_matches")
    def test_perform_pairwise_matching_no_inliers(
        self, mock_ransac_filter_matches, mock_compute_matches, mock_print
    ):
        interest_point_info = {
            (18, 0): {
                "loc": {
                    "data": np.array(
                        [
                            [468.345027587921, 488.36223951244153, 2.970786928172377],
                            [856.6701082186948, 416.01488311517676, 3.4227515981883694],
                        ]
                    ),
                    "num_items": 2,
                    "shape": (936, 3),
                }
            },
            (18, 1): {
                "loc": {
                    "data": np.array(
                        [
                            [468.345027587921, 488.36223951244153, 2.970786928172377],
                            [856.6701082186948, 416.01488311517676, 3.4227515981883694],
                        ]
                    ),
                    "num_items": 2,
                    "shape": (936, 3),
                }
            },
        }
        view_paths = {}
        all_matches = []
        labels = []
        method = "test_method"

        mock_compute_matches.return_value = [(0, 0), (1, 1)]
        mock_ransac_filter_matches.return_value = ([], None)

        perform_pairwise_matching(
            interest_point_info, view_paths, all_matches, labels, method
        )

        self.assertEqual(len(all_matches), 0)
        mock_print.assert_any_call("⚠️ No inlier matches found after RANSAC filtering.")

    # @unittest.skip
    @patch("builtins.print")
    @patch("xml.etree.ElementTree.parse")
    @patch("Rhapso.matching.interest_point_matching.parse_and_read_datasets")
    @patch("Rhapso.matching.interest_point_matching.perform_pairwise_matching")
    @patch("Rhapso.matching.interest_point_matching.save_matches_as_n5")
    def test_start_matching_local(
        self,
        mock_save_matches_as_n5,
        mock_perform_pairwise_matching,
        mock_parse_and_read_datasets,
        mock_et_parse,
        mock_print,
    ):
        mock_et_parse.return_value = MagicMock(
            getroot=MagicMock(return_value=ET.Element("root"))
        )
        mock_parse_and_read_datasets.return_value = (
            {
                (18, 30): {
                    "loc": {
                        "data": np.array([[1, 2, 3], [4, 5, 6]]),
                        "num_items": 2,
                        "shape": (936, 3),
                    }
                }
            },
            {
                (18, "0"): {
                    "timepoint": 18,
                    "path": "tpId_18_viewSetupId_0",
                    "setup": "0",
                }
            },
        )

        start_matching("/path/to/local/xml", "/path/to/n5", "/path/to/output")

        mock_print.assert_any_call("📂 Using local XML file: /path/to/local/xml")
        # mock_print.assert_any_call("📊 XML file contains 1 timepoints: [18]")
        mock_print.assert_any_call("\n📦 Collected Interest Point Info:")
        mock_print.assert_any_call(
            "\n✅ Successfully processed XML file with 1 timepoints defined in XML"
        )
        mock_print.assert_any_call(
            "✅ Successfully loaded interest points for 1 timepoints"
        )
        mock_print.assert_any_call(
            "✅ Generated and saved 0 matches across all loaded timepoints"
        )
        mock_print.assert_any_call("✅ Output saved to: /path/to/output")

    @patch("builtins.print")
    @patch("xml.etree.ElementTree.parse")
    @patch("Rhapso.matching.interest_point_matching.parse_and_read_datasets")
    @patch("Rhapso.matching.interest_point_matching.perform_pairwise_matching")
    @patch("Rhapso.matching.interest_point_matching.save_matches_as_n5")
    def test_start_matching_exception(
        self,
        mock_save_matches_as_n5,
        mock_perform_pairwise_matching,
        mock_parse_and_read_datasets,
        mock_et_parse,
        mock_print,
    ):
        mock_et_parse.side_effect = Exception("Test error")

        with self.assertRaises(SystemExit):
            start_matching("/path/to/local/xml", "/path/to/n5", "/path/to/output")

        mock_print.assert_any_call("❌ Error in start_matching function: Test error")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
