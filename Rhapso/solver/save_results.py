import xml.etree.ElementTree as ET
import boto3
import io
import s3fs
import json

"""
Utility class that saves the final matrices of alignment per view to XML
"""

class SaveResults:
    def __init__(self, tiles, xml_list, run_type, validation_stats):
        self.tiles = tiles[0]['tiles']
        self.xml_list = xml_list
        self.run_type = run_type
        self.validation_stats = validation_stats

        self.s3 = boto3.client("s3")
        self.s3fs = s3fs.S3FileSystem(anon=False)
        self.xml_file = None
        self.xml_file_path = None
        self.printed_views = set()
    
    def save_metrics(self, n5_input_path):
        path = n5_input_path + self.run_type + "_solver_metrics.txt"
        if n5_input_path.startswith("s3://"):
            with self.s3fs.open(path, "w", encoding="utf-8") as f:
                json.dump(self.validation_stats, f, default=str, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.validation_stats, f, default=str, indent=2)
    
    def save_xml(self, xml_file_path, tree):
        """
        Saves the XML tree to either an S3 bucket or the local filesystem based on the file source.
        """
        if xml_file_path.startswith("s3://"):
            xml_bytes = io.BytesIO()
            tree.write(xml_bytes, encoding='utf-8', xml_declaration=True)
            xml_bytes.seek(0)
            no_scheme = xml_file_path.replace("s3://", "", 1)
            bucket, key = no_scheme.split("/", 1)
            self.s3.upload_fileobj(xml_bytes, bucket, key)
        else:
            with open(xml_file_path, 'wb') as file:
                tree.write(file, encoding='utf-8', xml_declaration=True)
    
    def add_new_view_transform(self, root):
        """
        Adds an affine transform entry to each view in the XML, using fitted or default values.
        """
        for view_registration in root.findall('.//ViewRegistration'):
            timepoint = view_registration.get('timepoint')
            setup = view_registration.get('setup')
            view = f"timepoint: {timepoint}, setup: {setup}"

            new_view_transform = ET.Element('ViewTransform', {'type': 'affine'})
            new_view_transform.text = "\n\t\t\t" 
            
            name = ET.SubElement(new_view_transform, 'Name')
            if self.run_type == "rigid":
                name.text = 'RigidModel3D, lambda = 0.5'
            elif self.run_type == "affine" or self.run_type == "split-affine":
                name.text = 'AffineModel3D regularized with a RigidModel3D, lambda = 0.05'
            name.tail = "\n\t\t\t"

            affine = ET.SubElement(new_view_transform, 'affine')
                 
            tile = next((tile for tile in self.tiles if tile['view'] == view), None)
            model = (tile or {}).get('model', {}).get('regularized', {})
            
            if not model or all(float(v) == 0.0 for v in model.values()):
                affine.text = '1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0'
            else:
                affine.text = ' '.join(str(model.get(f'm{i}{j}', 0.0)) for i in range(3) for j in range(4))
                self.printed_views.add(f"tile: {view}, model: {affine.text}")
      
            view_registration.text = "\n\t\t\t"
            view_registration.insert(0, new_view_transform)
            new_view_transform.text = "\n\t\t\t\t"   
            name.tail               = "\n\t\t\t\t"   
            affine.tail             = "\n\t\t\t"    
            new_view_transform.tail = "\n\t\t\t"
        
        return root

    def load_xml(self, xml_file):
        """
        Parses the loaded XML string and initializes the ElementTree structure.
        """
        root = ET.fromstring(xml_file)
        tree = ET.ElementTree(root)

        return tree, root

    def run(self):
        """
        Executes the entry point of the script.
        """
        for xml in self.xml_list:
            xml_file = xml['xml_file']
            xml_file_path = xml['output_xml_path']
            n5_path = xml['n5_path']
            
            tree, root = self.load_xml(xml_file)
            root = self.add_new_view_transform(root)
            tree = ET.ElementTree(root)
            self.save_xml(xml_file_path, tree)
            # self.save_metrics(n5_path)
        
        for printed_view in self.printed_views:
            print(printed_view)