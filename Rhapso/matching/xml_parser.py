import xml.etree.ElementTree as ET
import boto3

"""
Utility class to parse XML content
"""

class XMLParser:
    def __init__(self, xml_input_path, input_type):
        """Initialize XML parser with file path or XML content string"""
        self.xml_input_path = xml_input_path
        self.data_global = None  # Will hold complete dataset info
        self.input_type = input_type
    
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

    def parse_view_setup(self, root):
        timepoints = set()

        # zarr loader 
        for zg in root.findall(".//ImageLoader/zgroups/zgroup"):
            tp_attr = zg.get("tp") or zg.get("timepoint") or "0"
            try:
                timepoints.add(int(tp_attr))
            except ValueError:
                pass

        # tiff loader 
        for fm in root.findall(".//ImageLoader/files/FileMapping"):
            tp_attr = fm.get("timepoint") or "0"
            try:
                timepoints.add(int(tp_attr))
            except ValueError:
                pass

        if not timepoints:
            timepoints = {0}

        # --- parse ViewSetups ---
        by_id = {}
        for vs in root.findall(".//ViewSetups/ViewSetup"):
            sid = int(vs.findtext("id"))
            name = (vs.findtext("name") or "").strip()

            size_txt = (vs.findtext("size") or "").strip()
            try:
                sx, sy, sz = [int(x) for x in size_txt.split()]
            except Exception:
                sx = sy = sz = None

            vox_txt = (vs.findtext("voxelSize/size") or "").strip()
            try:
                vx, vy, vz = [float(x) for x in vox_txt.split()]
            except Exception:
                vx = vy = vz = None

            attrs = {}
            attrs_node = vs.find("attributes")
            if attrs_node is not None:
                for child in list(attrs_node):
                    txt = (child.text or "").strip()
                    try:
                        attrs[child.tag] = int(txt)
                    except ValueError:
                        attrs[child.tag] = txt

            by_id[sid] = {
                "id": sid,
                "name": name,
                "size": (sx, sy, sz),
                "voxelSize": (vx, vy, vz),
                "attributes": attrs,
            }

        # --- expand to (timepoint, setup) maps for overlap checks ---
        viewSizes = {}
        viewVoxelSizes = {}
        for tp in sorted(timepoints):
            for sid, meta in by_id.items():
                if meta["size"] != (None, None, None):
                    viewSizes[(tp, sid)] = meta["size"]
                if meta["voxelSize"] != (None, None, None):
                    viewVoxelSizes[(tp, sid)] = meta["voxelSize"]

        return {
            "byId": by_id,
            "viewSizes": viewSizes,
            "viewVoxelSizes": viewVoxelSizes,
        }
    
    def parse_image_loader(self, root):
        image_loader_data = []
        
        if self.input_type == "zarr":       
            for il in root.findall(".//ImageLoader/zgroups/zgroup"):
                view_setup = il.get("setup")
                timepoint = il.get('timepoint') if 'timepoint' in il else il.get('tp')
                file_path = (il.get("path") or il.findtext("path") or "").strip()
                image_loader_data.append(
                    {
                        "view_setup": view_setup,
                        "timepoint": timepoint,
                        "series": 1,
                        "channel": 1,
                        "file_path": file_path,
                    }
                )
        elif self.input_type == "tiff":
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
                full_path = self.xml_input_path.replace("dataset-detection.xml", "") + file_path
                image_loader_data.append(
                    {
                        "view_setup": view_setup,
                        "timepoint": timepoint,
                        "series": series,
                        "channel": channel,
                        "file_path": full_path,
                    }
                )

        return image_loader_data

    def parse(self, xml_content):
        """Parse XML file or string and create complete dataset object"""
        try:
            # Check if the input is a string containing XML content
            if xml_content.strip().startswith('<?xml') or self.xml_file.strip().startswith('<'):
                # Parse XML from string content
                root = ET.fromstring(xml_content)
            else:
                # Parse XML from file path
                tree = ET.parse(xml_content)
                root = tree.getroot()
            
            # Create comprehensive data_global structure
            self.data_global = {
                'basePathURI': root.find(".//BasePath").text if root.find(".//BasePath") is not None else "",
                'viewRegistrations': self._parse_view_registrations(root),
                'viewsInterestPoints': self._parse_view_paths(root),
                'imageLoader': self.parse_image_loader(root),
                'viewSetup': self.parse_view_setup(root)
            }
            return self.data_global
            
        except Exception as e:
            print(f"❌ Error parsing XML content: {e}")
            raise

    def _parse_view_registrations(self, root):
        """Parse ViewRegistration entries from XML"""
        print("🔍 Parsing ViewRegistration entries...")
        
        view_registrations = {}
        
        # Find all ViewRegistration elements
        for view_reg in root.findall(".//ViewRegistration"):
            try:
                # Extract timepoint and setup
                timepoint = int(view_reg.get('timepoint'))
                setup = int(view_reg.get('setup'))
                view_id = (timepoint, setup)
                
                print(f"🔍 Processing ViewRegistration for view {view_id}")
                
                # Parse all ViewTransform elements for this view
                transforms = []
                for transform_elem in view_reg.findall("ViewTransform"):
                    transform_type = transform_elem.get('type', 'unknown')
                    
                    # Extract the Name element
                    name_elem = transform_elem.find('Name')
                    transform_name = name_elem.text.strip() if name_elem is not None and name_elem.text else f"Unnamed_{transform_type}"
                    
                    # Extract the affine transformation matrix
                    affine_elem = transform_elem.find('affine')
                    if affine_elem is not None and affine_elem.text:
                        affine_text = affine_elem.text.strip()
                        
                        transform_data = {
                            'type': transform_type,
                            'name': transform_name,
                            'affine': affine_text
                        }
                        transforms.append(transform_data)
                        print(f"  ✅ Added transform: type='{transform_type}', name='{transform_name}'")
                    else:
                        print(f"  ⚠️ No affine data found for transform type='{transform_type}', name='{transform_name}'")
                        pass
                
                if transforms:
                    view_registrations[view_id] = transforms
                    print(f"📝 Stored {len(transforms)} transforms for view {view_id}")
                else:
                    print(f"⚠️ No valid transforms found for view {view_id}")
                    pass
                    
            except Exception as e:
                print(f"❌ Error parsing ViewRegistration: {e}")
                continue
        
        print(f"Parsed {len(view_registrations)} ViewRegistration entries")
        return view_registrations

    def _parse_view_paths(self, root):
        """Parse view interest point file paths"""
        view_paths = {}
        for vip in root.findall(".//ViewInterestPointsFile"):
            setup_id = int(vip.attrib['setup']) 
            timepoint = int(vip.attrib['timepoint'])
            label = vip.attrib.get('label', 'beads') 
            params = vip.attrib.get('params', '')
            path = (vip.text or '').strip().split('/', 1)[0]
            
            key = (timepoint, setup_id)
            
            if key in view_paths and label not in view_paths[key]['label']:
                view_paths[key]['label'].append(label)
            else:
                view_paths[key] = {
                    'timepoint': timepoint, 
                    'setup': setup_id,
                    'label': [label],
                    'params': params,
                    'path': path
                }
            
        return view_paths
    
    def fetch_local_xml(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            print(f"pipeline failed, could not find xml file located at '{file_path}'")
            return None
        except Exception as e:
            print(f"pipeline failed, error while parsing xml file at '{file_path}': {e}")
            return None
    
    def get_xml_content(self):
        """
        Fetches XML content from either S3 or local filesystem based on path prefix.
        Returns (xml_content, interest_points_folder) or (None, None) if not found.
        """
        # Determine the directory and interest points folder based on path type
        if self.xml_input_path.startswith('s3://'):
            # Parse S3 URL components
            s3_path = self.xml_input_path[5:]  
            parts = s3_path.split('/', 1)
            bucket_name = parts[0]
            file_key = parts[1]
            
            print(f"Detected S3 path. Fetching from bucket: {bucket_name}, key: {file_key}")
            
            # Initialize S3 client and fetch content
            s3_client = boto3.client('s3')
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            xml_content = response["Body"].read().decode("utf-8")
            
        else:
            print(f"Detected local path: {self.xml_input_path}")
            xml_content = self.fetch_local_xml(self.xml_input_path)
            if xml_content is None:
                return None, None
        
        return xml_content
    
    def run(self):
        xml_content = self.get_xml_content()
        data_global = self.parse(xml_content)
        return data_global

