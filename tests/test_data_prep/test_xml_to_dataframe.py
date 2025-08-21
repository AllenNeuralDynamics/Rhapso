import unittest
import xml.etree.ElementTree as ET
from Rhapso.data_prep.xml_to_dictionary import XMLToDictionary

class TestXMLToDictionary(unittest.TestCase):

    def setUp(self):
        self.xml_content_standard = "tests/XML_test_data/dataset.xml"
        self.xml_content_no_tags = "tests/XML_test_data/dataset_no_tags.xml"
        self.xml_content_interestPoints = (
            "tests/XML_test_data/dataset_interest_points.xml"
        )
        self.xml_content_unequal_lengths = (
            "tests/XML_test_data/dataset_unequal_lengths.xml"
        )
        self.xml_content_no_file_mapping = (
            "tests/XML_test_data/dataset_no_file_mapping.xml"
        )

    def fetch_local_xml(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def test_parse_image_loader_tiff(self):
        xml_content = self.fetch_local_xml(self.xml_content_standard)
        self.parser = XMLToDictionary(xml_content)
        root = ET.fromstring(xml_content)
        df = self.parser.parse_image_loader_tiff(root)
        self.assertEqual(len(df), 14)
        self.assertEqual(df[0]["view_setup"], "5")
        self.assertEqual(df[0]["timepoint"], "30")
        self.assertEqual(df[0]["series"], "0")
        self.assertEqual(df[0]["channel"], "0")
        self.assertEqual(df[0]["file_path"], "spim_TL30_Angle225.tif")

    def test_parse_view_setups(self):
        xml_content = self.fetch_local_xml(self.xml_content_standard)
        self.parser = XMLToDictionary(xml_content)
        root = ET.fromstring(xml_content)
        df = self.parser.parse_view_setups(root)
        self.assertEqual(len(df), 7)
        self.assertEqual(df[0]["id"], "0")
        self.assertEqual(df[0]["name"], "0")
        self.assertEqual(df[0]["size"], "1388 1040 81")
        self.assertEqual(df[0]["voxel_unit"], "µm")
        self.assertEqual(
            df[0]["voxel_size"], "0.7310780550106993 0.7310780550106993 2.0"
        )

    def test_parse_view_registrations(self):
        xml_content = self.fetch_local_xml(self.xml_content_standard)
        self.parser = XMLToDictionary(xml_content)
        root = ET.fromstring(xml_content)
        df = self.parser.parse_view_registrations(root)
        self.assertEqual(len(df), 28)
        self.assertEqual(df[0]["timepoint"], "18")
        self.assertEqual(df[0]["setup"], "0")
        self.assertEqual(df[0]["type"], "affine")
        self.assertEqual(
            df[0]["name"], "Rotation around axis (1.0, 0.0, 0.0) by 0.0 degrees"
        )
        self.assertEqual(
            df[0]["affine"],
            "1.0, 0.0, 0.0, -694.0, 0.0, 1.0, 0.0, -520.0, 0.0, 0.0, 1.0, -110.79528300000001",
        )

    def test_parse_view_interest_points(self):
        xml_content = self.fetch_local_xml(self.xml_content_standard)
        self.parser = XMLToDictionary(xml_content)
        root = ET.fromstring(xml_content)
        df = self.parser.parse_view_interest_points(root)
        self.assertEqual(df, [])

    def test_run(self):
        xml_content = self.fetch_local_xml(self.xml_content_standard)
        self.parser = XMLToDictionary(xml_content)
        result = self.parser.run()
        self.assertIn("image_loader", result)
        self.assertIn("view_setups", result)
        self.assertIn("view_registrations", result)
        self.assertIn("view_interest_points", result)

    def test_no_labels(self):
        xml_content = self.fetch_local_xml(self.xml_content_no_tags)
        self.parser = XMLToDictionary(xml_content)
        root = ET.fromstring(xml_content)
        self.assertFalse(self.parser.check_labels(root))

    def test_raise_no_label(self):
        xml_content = self.fetch_local_xml(self.xml_content_no_tags)
        root = ET.fromstring(xml_content)
        self.parser = XMLToDictionary(xml_content)
        with self.assertRaises(Exception) as context:
            self.parser.parse_image_loader_tiff(root)
        self.assertEqual(str(context.exception), "Required labels do not exist")

    def test_check_length(self):
        xml_content = self.fetch_local_xml(self.xml_content_unequal_lengths)
        self.parser = XMLToDictionary(xml_content)
        root = ET.fromstring(xml_content)
        result = self.parser.check_length(root)
        self.assertFalse(result)

    def test_raise_bad_length(self):
        xml_content = self.fetch_local_xml(self.xml_content_unequal_lengths)
        root = ET.fromstring(xml_content)
        self.parser = XMLToDictionary(xml_content)
        with self.assertRaises(Exception) as context:
            self.parser.parse_image_loader_tiff(root)
        self.assertEqual(
            str(context.exception),
            "The amount of view setups, view registrations, and tiles do not match",
        )

    def test_no_file_mapping_exists(self):
        xml_content = self.fetch_local_xml(self.xml_content_no_file_mapping)
        self.parser = XMLToDictionary(xml_content)
        root = ET.fromstring(xml_content)
        with self.assertRaises(Exception) as context:
            self.parser.parse_image_loader_tiff(root)
        self.assertEqual(str(context.exception), "There are no files in this XML")


if __name__ == "__main__":
    unittest.main()
