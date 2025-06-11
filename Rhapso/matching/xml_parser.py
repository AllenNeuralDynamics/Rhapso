import xml.etree.ElementTree as ET
import numpy as np

class XMLParser:
    def __init__(self, xml_file):
        """Initialize XML parser with file path or XML content string"""
        self.xml_file = xml_file
        self.data_global = None  # Will hold complete dataset info

    def parse(self):
        """Parse XML file or string and create complete dataset object"""
        try:
            # Check if the input is a string containing XML content
            if self.xml_file.strip().startswith('<?xml') or self.xml_file.strip().startswith('<'):
                # Parse XML from string content
                root = ET.fromstring(self.xml_file)
            else:
                # Parse XML from file path
                tree = ET.parse(self.xml_file)
                root = tree.getroot()
            
            # Create comprehensive data_global structure
            self.data_global = {
                'basePathURI': root.find(".//BasePath").text if root.find(".//BasePath") is not None else "",
                'boundingBoxes': self._parse_bounding_boxes(root),
                'viewRegistrations': self._parse_view_registrations(root),
                'viewsInterestPoints': self._parse_view_paths(root),
                'sequenceDescription': self._parse_sequence_description(root)
            }
            return self.data_global
            
        except Exception as e:
            print(f"❌ Error parsing XML content: {e}")
            raise

    def _parse_sequence_description(self, root):
        """Parse sequence metadata from XML root"""
        # TODO: Implement parsing of sequence metadata
        return {}

    def _parse_bounding_boxes(self, root):
        """Parse bounding box information from XML"""
        # TODO: Implement bounding box parsing
        return {}

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
                
                if transforms:
                    view_registrations[view_id] = transforms
                    print(f"📝 Stored {len(transforms)} transforms for view {view_id}")
                else:
                    print(f"⚠️ No valid transforms found for view {view_id}")
                    
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

    def parse_timepoints(self):
        """Extract timepoint information from XML"""
        root = ET.parse(self.xml_file).getroot()
        timepoints = set()
        timepoints_list = root.find(".//Timepoints[@type='list']")
        if timepoints_list:
            for timepoint in timepoints_list.findall("Timepoint"):
                timepoints.add(int(timepoint.find("id").text))
        else:
            timepoint_range = root.find(".//Timepoints[@type='range']")
            if timepoint_range:
                first = int(timepoint_range.find("first").text)
                last = int(timepoint_range.find("last").text)
                timepoints = set(range(first, last + 1))
        return timepoints

    def build_label_map(self):
        """Build global label map for all views"""
        root = ET.parse(self.xml_file).getroot()
        label_map_global = {}
        label_weights = {}
        # TODO: Implement label mapping similar to BigStitcher
        return label_map_global, label_weights

    def setup_groups(self, view_registrations):
        """Set up view groups for pairwise matching"""
        # Get all views from viewsInterestPoints
        views = list(self.data_global['viewsInterestPoints'].keys())
        
        # Group views by timepoint
        timepoint_groups = {}
        for view in views:
            timepoint, setup_id = view
            if timepoint not in timepoint_groups:
                timepoint_groups[timepoint] = []
            timepoint_groups[timepoint].append(view)

        # Create pairs within each timepoint
        pairs = []
        for timepoint, timepoint_views in timepoint_groups.items():
            for i in range(len(timepoint_views)):
                for j in range(i + 1, len(timepoint_views)):
                    pairs.append((timepoint_views[i], timepoint_views[j]))

        print(f"Created {len(pairs)} pairs for matching")
        for pair in pairs[:5]:  # Print first 5 pairs as example
            print(f"Pair: ViewId[{pair[0][0]},{pair[0][1]}] <=> ViewId[{pair[1][0]},{pair[1][1]}]")
        if len(pairs) > 5:
            print("... and more pairs")

        return {
            'groups': timepoint_groups,
            'pairs': pairs,
            'rangeComparator': None,
            'subsets': None,
            'views': views
        }

    def get_data_global(self):
        """Get complete dataset information object"""
        if self.data_global is None:
            self.parse()
        return self.data_global
