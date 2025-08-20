import xml.etree.ElementTree as ET

# This component recieves an XML file containing Tiff or Zarr image metadata and converts
# it into several Dictionaries

class XMLToDictionary:
    def __init__(self, xml_file):
        self.xml_content = xml_file

    def parse_image_loader_zarr(self, root):
        """
        Parses image loader configuration from a Zarr file's XML structure and constructs a Dictionary containing the 
        metadata for each image group.
        """
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

        return image_loader_data

    def parse_image_loader_tiff(self, root):
        """
        Parses image loader configuration from a TIFF file's XML structure and constructs a Dictionary containing 
        metadata for each image group.
        """
        image_loader_data = []

        # Ensure that file mappings are present in the XML
        if not root.findall(".//ImageLoader/files/FileMapping"):
            raise Exception("There are no files in this XML")
        
        # Check for required labels in the XML
        if not self.check_labels(root):
            raise Exception("Required labels do not exist")

        # Validate that the lengths of view setups, registrations, and tiles match
        if not self.check_length(root):
            raise Exception(
                "The amount of view setups, view registrations, and tiles do not match"
            )

        # Iterate over each file mapping in the XML
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

        # Convert the list to a Dictionary and return
        return image_loader_data
    
    def parse_image_loader_split_zarr(self):
        pass

    def route_image_loader(self, root):
        """
        Directs the XML parsing process based on the image loader format specified in the XML.
        """
        format_node = root.find(".//ImageLoader")
        format_type = format_node.get("format")

        if "filemap" in format_type:
            return self.parse_image_loader_tiff(root)
        else:
            return self.parse_image_loader_zarr(root)

    def parse_view_setups(self, root):
        """
        Parses the view setups from an XML structure and constructs a Dictionary containing metadata for each view setup.
        """
        viewsetups_data = []

        for vs in root.findall(".//ViewSetup"):
            id_ = vs.find("id").text
            # name = vs.find("name").text
            name = vs.findtext("name")
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
        return viewsetups_data

    def parse_view_registrations(self, root):
        """
        Parses view registrations from an XML structure and constructs a Dictionary containing registration metadata 
        for each view.
        """
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
        return viewregistrations_data

    def parse_view_interest_points(self, root):
        """
        Parses interest points data from an XML structure and constructs a Dictionary containing metadata and paths 
        for each set of interest points.
        """
        view_interest_points_data = []

        # if self.key == "detection":
        #     if len(root.findall(".//ViewInterestPointsFile")) != 0:
        #         raise Exception("There should be no interest points in this file yet.")

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
        return view_interest_points_data

    def check_labels(self, root):
        """
        Verifies the presence of required XML labels including bounding boxes, point spread functions, 
        stitching results, and intensity adjustments.
        """
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
        """
        Validates that the count of elements within the XML structure aligns with expected relationships
        between file mappings, view setups, and view registrations.
        """
        length = True
        if len(root.findall(".//ImageLoader/files/FileMapping")) != len(root.findall(".//ViewRegistration")) or \
            len(root.findall(".//ViewSetup")) != len(root.findall(".//ViewRegistration")) * (1 / 2):
            length = False  # Set to False if the relationships do not match expected counts
        return length

    def run(self):
        """
        Executes the entry point of the script.
        """
        root = ET.fromstring(self.xml_content)
        image_loader = self.route_image_loader(root)
        view_setups = self.parse_view_setups(root)
        view_registrations = self.parse_view_registrations(root)
        view_interest_points = self.parse_view_interest_points(root)
        
        return {
            "image_loader": image_loader,
            "view_setups": view_setups,
            "view_registrations": view_registrations,
            "view_interest_points": view_interest_points,
        }
