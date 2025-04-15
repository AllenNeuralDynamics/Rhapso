from idna import check_label
import pandas as pd
import xml.etree.ElementTree as ET

# This component recieves an XML file containing Tiff or Zarr image metadata and converts
# it into several Dataframes


class XMLToDataFrame:
    def __init__(self, xml_file, key):
        self.xml_content = xml_file
        self.key = key

    def parse_image_loader_zarr(self, root):
        image_loader_data = []

        for il in root.findall(".//ImageLoader/zgroups/zgroup"):
            view_setup = il.get("setup")
            timepoint = il.get("timepoint")
            file_path = il.find("path").text if il.find("path") is not None else None

            image_loader_data.append(
                {
                    "view_setup": view_setup,
                    "timepoint": timepoint,
                    "series": 1,
                    "channel": 1,
                    "file_path": file_path,
                }
            )

        return pd.DataFrame(image_loader_data)

    def parse_image_loader_tiff(self, root):
        image_loader_data = []

        if (
            not root.findall(".//ImageLoader/files/FileMapping")
            or len(root.findall(".//ImageLoader/files/FileMapping")) == 0
        ):
            raise Exception("There are no files in this XML")
        if not self.check_labels(root):
            raise Exception("Required labels do not exist")
        if not self.check_length(root):
            raise Exception(
                "The amount of view setups, view registrations, and tiles do not match"
            )

        for fm in root.findall(".//ImageLoader/files/FileMapping"):
            view_setup = fm.get("view_setup")
            timepoint = fm.get("timepoint")
            series = fm.get("series")
            channel = fm.get("channel")
            file_path = fm.find("file").text if fm.find("file") is not None else None

            image_loader_data.append(
                {
                    "view_setup": view_setup,
                    "timepoint": timepoint,
                    "series": series,
                    "channel": channel,
                    "file_path": file_path,
                }
            )
        return pd.DataFrame(image_loader_data)

    def route_image_loader(self, root):
        format_node = root.find(".//ImageLoader")
        format_type = format_node.get("format")

        if "zarr" in format_type:
            return self.parse_image_loader_zarr(root)
        elif "filemap" in format_type:
            return self.parse_image_loader_tiff(root)
        else:
            raise ValueError("Unsupported format type")

    def parse_view_setups(self, root):
        viewsetups_data = []

        for vs in root.findall(".//ViewSetup"):
            id_ = vs.find("id").text
            name = vs.find("name").text
            size = vs.find("size").text
            voxel_unit = vs.find(".//voxelSize/unit").text
            voxel_size = " ".join(vs.find(".//voxelSize/size").text.split())
            attributes = {attr.tag: attr.text for attr in vs.find("attributes")}
            viewsetups_data.append(
                {
                    "id": id_,
                    "name": name,
                    "size": size,
                    "voxel_unit": voxel_unit,
                    "voxel_size": voxel_size,
                    **attributes,
                }
            )
        return pd.DataFrame(viewsetups_data)

    def parse_view_registrations(self, root):
        viewregistrations_data = []
        for vr in root.findall(".//ViewRegistration"):
            timepoint = vr.get("timepoint")
            setup = vr.get("setup")

            for vt in vr.findall(".//ViewTransform"):
                affine_text = (
                    vt.find("affine").text.replace("\n", "").replace(" ", ", ")
                )
                viewregistrations_data.append(
                    {
                        "timepoint": timepoint,
                        "setup": setup,
                        "type": vt.get("type"),
                        "name": vt.find("Name").text.strip(),
                        "affine": affine_text,
                    }
                )
        return pd.DataFrame(viewregistrations_data)

    def parse_view_interest_points(self, root):
        view_interest_points_data = []

        if self.key == "detection":
            if len(root.findall(".//ViewInterestPointsFile")) != 0:
                raise Exception("There should be no interest points in this file yet.")

        for vip in root.findall(".//ViewInterestPointsFile"):
            timepoint = vip.get("timepoint")
            setup = vip.get("setup")
            label = vip.get("label")
            params = vip.get("params")
            path = vip.text.strip() if vip.text is not None else None
            view_interest_points_data.append(
                {
                    "timepoint": timepoint,
                    "setup": setup,
                    "label": label,
                    "params": params,
                    "path": path,
                }
            )
        return pd.DataFrame(view_interest_points_data)

    def check_labels(self, root):
        labels = True
        if root.find(".//BoundingBoxes") is None:
            labels = False
        if root.find(".//PointSpreadFunctions") is None:
            labels = False
        if root.find(".//StitchingResults") is None:
            labels = False
        if root.find(".//IntensityAdjustments") is None:
            labels = False

        return labels

    def check_length(self, root):
        length = True
        if len(root.findall(".//ImageLoader/files/FileMapping")) != len(
            root.findall(".//ViewRegistration")
        ) and len(root.findall(".//ViewSetup")) != len(
            root.findall(".//ViewRegistration")
        ) * (
            1 / 2
        ):
            length = False
        return length

    def run(self):
        root = ET.fromstring(self.xml_content)
        image_loader_df = self.route_image_loader(root)
        view_setups_df = self.parse_view_setups(root)
        view_registrations_df = self.parse_view_registrations(root)
        view_interest_points_df = self.parse_view_interest_points(root)

        return {
            "image_loader": image_loader_df,
            "view_setups": view_setups_df,
            "view_registrations": view_registrations_df,
            "view_interest_points": view_interest_points_df,
        }
