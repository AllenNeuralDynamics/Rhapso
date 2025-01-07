import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Set, Tuple
from io import BytesIO
import boto3
from config import Config
import numpy as np

class SequenceDescription:
    def __init__(self, root, s3):
        self.root = root
        self.s3 = s3
    
    def view_setups(self) -> Dict[int, Any]:
        view_setup_elements = self.root.find(Config.VIEWSETUPSXPATH)
        result = {}
        
        for elem in view_setup_elements:
            child = elem.find("id")
            if child is not None and child.text is not None:
                id_int = int(child.text)
                result[id_int] = elem
        
        return result

    def time_points(self) -> Dict[str, Any]:
        timepoints_elem = self.root.find(Config.TIMEPOINTSXPATH)
        if timepoints_elem is None: return {}
        timepoints_type = timepoints_elem.get("type")
        result = {}
        
        if timepoints_type == "pattern":
            pattern_elem = timepoints_elem.find('integerpattern')
            if pattern_elem is not None:
                pattern_text = pattern_elem.text
                result = {'pattern': pattern_text}
            else:
                raise ValueError("Pattern tag missing.")
            
        elif timepoints_type == "range":
            first = int(timepoints_elem.find('first').text)
            last = int(timepoints_elem.find('last').text)
            range_map = {t: {'timepoint': t} for t in range(first, last + 1)}
            result = {'range': range_map}

        elif timepoints_type == "list":
            list_map = {}
            for tp_elem in timepoints_elem:
                tp_id_elem = tp_elem.find('id')
                if tp_id_elem is not None and tp_id_elem.text is not None:
                    tp_id = int(tp_id_elem.text)
                    list_map[tp_id] = {'id': tp_id}
                else:
                    raise ValueError("ID tag is missing.")
            result = {'list': list_map}
        
        else:
            raise ValueError(f"Unknown timepoints type: {timepoints_type}")

        return result
    
    def missing_views(self) -> Set[Tuple[int, int]]:
        missing_views_elem = self.root.find(Config.MISSINGVIEWSXPATH)
        if missing_views_elem is None: return set()
        views = set()

        for elem in missing_views_elem: 
            timepoint = int(elem.get('timepoint')) 
            setup = int(elem.get('setup'))  
            views.add((timepoint, setup)) 

        return views
    
    def image_loader(self, sequence_description) -> Dict[str, Any]:
        zgroups_elem = self.root.find(Config.ZGROUPSXPATH)
        if zgroups_elem is None: return {}
        zgroups = {}

        for c in zgroups_elem.findall("zgroup"):
            timepoint_id = int(c.get("timepoint"))
            setup_id = int(c.get("setup"))
            path = c.find("path").text if c.find("path") is not None else None
            zgroups[(timepoint_id, setup_id)] = path

        keyValueReader = {
            'multi_scale_path': Config.BASE_PATH,
            's3_client': self.s3,
            'bucket_name': Config.BUCKET_NAME,
            's3_mode': self.s3 is not None
        }

        zarrImageLoader = {
            'z_groups' : zgroups,
            'seq' : sequence_description,
            'zarr_key_value_reader_builder' : keyValueReader
        }

        return zarrImageLoader
    
    def run(self) -> Dict[str, Dict[str, Any]]:
        sequence_description = {}
        sequence_description['view_setups'] = self.view_setups()
        sequence_description['time_points'] = self.time_points()
        sequence_description['missing_views'] = self.missing_views()
        sequence_description['image_loader'] = self.image_loader(sequence_description)

        return sequence_description

class ViewRegistrations: 
    def __init__(self, root):
        self.root = root
    
    def view_registrations(self):
        view_registrations_elems = self.root.find(Config.VIEWREGISTRATIONSXPATH)
        if view_registrations_elems is None: raise ValueError("View registrations element not found")
        registrations = {}
        
        for elem in view_registrations_elems:
            timepoint_id = int(elem.get('timepoint'))  
            setup_id = int(elem.get('setup'))
            transforms = []

            transforms_elems = elem.findall('ViewTransform')
            for t_elem in transforms_elems:
                view_type = t_elem.get('type')

                if view_type == 'affine':
                    name = t_elem.find('Name').text if t_elem.find('Name') is not None else None
                    affine_data = t_elem.find('affine').text if t_elem.find('affine') is not None else None
                    
                    if affine_data:
                        affine_matrix = np.array(list(map(float, affine_data.split())))
                        if affine_matrix.size == 12:  # Ensuring it's a 3D affine transform (4x3 matrix)
                            affine_matrix = affine_matrix.reshape((3, 4))
                        else:
                            raise ValueError("Invalid affine transform data")
                    
                    transforms.append({'type': 'affine', 'name': name, 'affine': affine_matrix.tolist()})

                elif view_type == 'generic':
                    attribute_class_name = t_elem.get('class')
                    transforms.append({'type': 'generic', 'class': attribute_class_name})
                
                else:
                    raise ValueError(f"Unknown <{'ViewTransform'}> type: {view_type}")
        
        registrations[(timepoint_id, setup_id)] = {
            'timepoint_id': timepoint_id,
            'setup_id': setup_id,
            'transforms': transforms
        }

        return registrations

    def run(self):
        return self.view_registrations() 
    
class ViewInterestPoints:
    def __init__(self, root):
        self.root = root
    
    def view_interest_points(self):
        view_interest_points_elems = self.root.find(Config.VIEWINTERESTPOINTSXPATH)
        view_interest_points = {}
        if view_interest_points_elems is None:
            return view_interest_points
        
        interestPointCollectionLookup = {}

        for elem in view_interest_points_elems:
            timepoint_id = int(elem.get('timepoint')) 
            setup_id = int(elem.get('setup'))
            label = elem.get('label')
            parameters = elem.get('params')
            interest_point_file_name = elem.text.strip() 

            view_id = (timepoint_id, setup_id)
            
            if view_id not in interestPointCollectionLookup:
                interestPointCollectionLookup[view_id] = {
                    'timepoint_id': timepoint_id,
                    'setup_id': setup_id
                }
            
            collection = interestPointCollectionLookup[view_id]



            return view_interest_points

    def run(self):
        return self.view_interest_points()

class InputPreparation:
    def __init__(self):
        self.bucket_name = Config.BUCKET_NAME
        self.s3 = boto3.client('s3', region_name=Config.REGION)
    
    def fetch_xml_data(self, file_key: str):
        xml_file = BytesIO()
        self.s3.download_fileobj(self.bucket_name, file_key, xml_file)
        xml_file.seek(0)  
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return root
    
    def run(self) -> Dict[str, Any]:
        root = self.fetch_xml_data(Config.XMLFILENAME)

        # We might need to dynamically check for missing XML tags to ensure completeness of parsing. 
        # This involves starting with a list of expected tags and for each new parsing method, checking for new 
        # tags at each level of the XML structure beginning from the root. This process, while not currently 
        # implemented, would be initiated here once its impact is evaluated.

        from_sequence_description = SequenceDescription(root, self.s3)
        from_view_registrations = ViewRegistrations(root)
        from_view_interest_points = ViewInterestPoints(root)

        spim_data = {
            'sequence_description': from_sequence_description.run(),
            'view_registrations' : from_view_registrations.run(),
            'view_interest_points' : from_view_interest_points.run()
        }

        return spim_data


