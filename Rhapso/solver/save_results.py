import xml.etree.ElementTree as ET
import boto3
import io
import s3fs
import json

"""
Utility class that saves the final matrices of alignment per view to XML
"""

class SaveResults:
    def __init__(self, tiles, xml_file, xml_file_path, run_type, validation_stats, n5_input_path):
        self.tiles = tiles
        self.xml_file = xml_file
        self.xml_file_path = xml_file_path
        self.run_type = run_type
        self.validation_stats = validation_stats
        self.n5_input_path = n5_input_path
    
    def save_ransac_metrics(self):
        path = self.n5_input_path + self.run_type + "-solver_metrics.txt"
        if self.n5_input_path.startswith("s3://"):
            fs = s3fs.S3FileSystem(anon=False)
            with fs.open(path, "w", encoding="utf-8") as f:
                json.dump(self.validation_stats, f, default=str, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.validation_stats, f, default=str, indent=2)
    
    def save_xml(self):
        """
        Saves the XML tree to either an S3 bucket or the local filesystem based on the file source.
        """
        if self.xml_file_path.startswith("s3://"):
            xml_bytes = io.BytesIO()
            self.tree.write(xml_bytes, encoding='utf-8', xml_declaration=True)
            xml_bytes.seek(0)
            no_scheme = self.xml_file_path.replace("s3://", "", 1)
            bucket, key = no_scheme.split("/", 1)
            s3 = boto3.client("s3")
            s3.upload_fileobj(xml_bytes, bucket, key)
        else:
            with open(self.xml_file_path, 'wb') as file:
                self.tree.write(file, encoding='utf-8', xml_declaration=True)
    
    def add_new_view_transform(self):
        """
        Adds an affine transform entry to each view in the XML, using fitted or default values.
        """
        for view_registration in self.root.findall('.//ViewRegistration'):
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
                print(f"tile: {view}, model: {affine.text}")
      
            view_registration.text = "\n\t\t\t"
            view_registration.insert(0, new_view_transform)
            new_view_transform.text = "\n\t\t\t\t"   
            name.tail               = "\n\t\t\t\t"   
            affine.tail             = "\n\t\t\t"    
            new_view_transform.tail = "\n\t\t\t"

    def load_xml(self):
        """
        Parses the loaded XML string and initializes the ElementTree structure.
        """
        self.root = ET.fromstring(self.xml_file)
        self.tree = ET.ElementTree(self.root)

    def run(self):
        """
        Executes the entry point of the script.
        """
        self.load_xml()
        self.add_new_view_transform()
        self.save_xml()
        self.save_ransac_metrics()