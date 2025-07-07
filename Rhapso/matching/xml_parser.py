import xml.etree.ElementTree as ET
import boto3
import os

class XMLParser:
    def __init__(self, xml_input_path):
        """Initialize XML parser with file path or XML content string"""
        self.xml_input_path = xml_input_path
        self.data_global = None  # Will hold complete dataset info

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
                for transform_elem in view_reg.findall(".//ViewTransform"):
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
            # Parse attributes correctly - setup is a string, convert timepoint to int
            setup_id = int(vip.attrib['setup'])  # Convert setup to int for consistency
            timepoint = int(vip.attrib['timepoint'])
            label = vip.attrib.get('label', 'beads')  # Default to 'beads' if not specified
            params = vip.attrib.get('params', '')
            
            # Get the path text and clean it
            path = vip.text.strip()
            
            # Remove /beads suffix if present to get base path
            if path.endswith("/beads"):
                path = path[:-len("/beads")]
            
            # Create key as tuple (timepoint, setup_id)
            key = (timepoint, setup_id)
            
            # Store comprehensive view information
            view_paths[key] = {
                'timepoint': timepoint, 
                'setup': setup_id,
                'label': label,
                'params': params,
                'path': path
            }
            
        print(f"Parsed {len(view_paths)} ViewInterestPointsFile entries")
        for key, info in list(view_paths.items())[:3]:  # Show first 3 entries
            print(f"  View {key}: timepoint={info['timepoint']}, setup={info['setup']}, label={info['label']}")
        if len(view_paths) > 3:
            print(f"  ... and {len(view_paths) - 3} more views")
            
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
            s3_path = self.xml_input_path[5:]  # Remove 's3://'
            parts = s3_path.split('/', 1)
            bucket_name = parts[0]
            file_key = parts[1]
            
            print(f"Detected S3 path. Fetching from bucket: {bucket_name}, key: {file_key}")
            
            # Initialize S3 client and fetch content
            s3_client = boto3.client('s3')
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            xml_content = response["Body"].read().decode("utf-8")
            
            # Create S3 path for interest points folder
            xml_dir = os.path.dirname(file_key)
            interest_points_folder = f"s3://{bucket_name}/{xml_dir}/interestpoints.n5"
        else:
            print(f"Detected local path: {self.xml_input_path}")
            xml_content = self.fetch_local_xml(self.xml_input_path)
            if xml_content is None:
                return None, None
            # Create local path for interest points folder
            xml_dir = os.path.dirname(self.xml_input_path)
            interest_points_folder = os.path.join(xml_dir, 'interestpoints.n5')
        
        return xml_content, interest_points_folder
    
    def run(self):
        xml_content, interest_points_folder = self.get_xml_content()
        data_global = self.parse(xml_content)
        return data_global, interest_points_folder

