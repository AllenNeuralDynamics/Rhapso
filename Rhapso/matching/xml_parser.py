import xml.etree.ElementTree as ET
import numpy as np

class XMLParser:
    def __init__(self, xml_file):
        """Initialize XML parser with file path"""
        self.xml_file = xml_file
        self.data_global = None  # Will hold complete dataset info

    def parse(self):
        """Parse XML file and create complete dataset object"""
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        
        # Create comprehensive data_global structure
        self.data_global = {
            'basePathURI': root.find(".//BasePath").text,
            'boundingBoxes': self._parse_bounding_boxes(root),
            'viewRegistrations': self._parse_view_registrations(root),
            'viewsInterestPoints': self._parse_view_paths(root),
            'sequenceDescription': self._parse_sequence_description(root)
        }
        return self.data_global

    def _parse_sequence_description(self, root):
        """Parse sequence metadata from XML root"""
        # TODO: Implement parsing of sequence metadata
        return {}

    def _parse_bounding_boxes(self, root):
        """Parse bounding box information from XML"""
        # TODO: Implement bounding box parsing
        return {}

    def _parse_view_registrations(self, root):
        """Parse view registration parameters and transformations"""
        registrations = {}
        for reg in root.findall(".//ViewRegistration"):
            timepoint = int(reg.attrib['timepoint'])
            setup_id = int(reg.attrib['setup'])
            view_key = (timepoint, setup_id)
            
            # Find ViewTransform elements
            transforms = []
            for transform in reg.findall(".//ViewTransform"):
                transform_type = transform.attrib.get('type', 'unknown')
                
                # Parse affine transformation matrix
                if transform_type == 'affine':
                    affine_elem = transform.find('affine')
                    if affine_elem is not None:
                        matrix_text = affine_elem.text.strip()
                        # Parse matrix values (typically space or comma separated)
                        matrix_values = [float(x) for x in matrix_text.replace(',', ' ').split()]
                        
                        # Reshape to 3x4 matrix (typical for affine transforms)
                        if len(matrix_values) == 12:
                            matrix_3x4 = np.array(matrix_values).reshape(3, 4)
                            # Convert to 4x4 homogeneous matrix
                            matrix_4x4 = np.eye(4)
                            matrix_4x4[:3, :] = matrix_3x4
                        else:
                            # Default to identity if parsing fails
                            matrix_4x4 = np.eye(4)
                        
                        transforms.append({
                            'type': transform_type,
                            'matrix': matrix_4x4
                        })
            
            registrations[view_key] = {
                'timepoint': timepoint,
                'setup': setup_id,
                'transforms': transforms
            }
            
        print(f"Parsed {len(registrations)} ViewRegistration entries")
        return registrations

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
