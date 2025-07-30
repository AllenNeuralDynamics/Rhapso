import xml.etree.ElementTree as ET
import boto3
import io

# This class appends the transformation matrices to the existing XML file

class SaveResults:
    def __init__(self, tiles, xml_file, xml_bucket_name, xml_file_path, fixed_views, file_source):
        self.tiles = tiles
        self.xml_file = xml_file
        self.xml_bucket_name = xml_bucket_name
        self.xml_file_path = xml_file_path
        self.fixed_views = fixed_views
        self.file_source = file_source
    
    def save_xml(self):
        """
        Saves the XML tree to either an S3 bucket or the local filesystem based on the file source.
        """
        if self.file_source == 's3':
            xml_bytes = io.BytesIO()
            self.tree.write(xml_bytes, encoding='utf-8', xml_declaration=True)
            xml_bytes.seek(0)
            s3 = boto3.client('s3')
            s3.upload_fileobj(xml_bytes, self.xml_bucket_name, self.xml_file_path)
        elif self.file_source == 'local':
            with open(self.xml_file_path, 'wb') as file:
                self.tree.write(file, encoding='utf-8', xml_declaration=True)
        else:
            print("Invalid file source")
            return
    
    def add_new_view_transform(self):
        """
        Adds an affine transform entry to each view in the XML, using fitted or default values.
        """
        for view_registration in self.root.findall('.//ViewRegistration'):
            timepoint = view_registration.get('timepoint')
            setup = view_registration.get('setup')
            view = f"timepoint: {timepoint}, setup: {setup}"
            new_view_transform = ET.Element('ViewTransform', {'type': 'affine'})
            name = ET.SubElement(new_view_transform, 'Name')
            name.text = 'Interpolated3DAffine'
            affine = ET.SubElement(new_view_transform, 'affine')
            
            if view in self.fixed_views:
                affine.text = '1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0'
            else:
                model = self.tiles.get(view, {}).get('model', {}).get('a', {})
                if not model or all(float(val) == 0.0 for key, val in model.items()):
                    affine.text = '1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0'
                else:   
                    affine.text = ' '.join(f"{model.get(f'm{i}{j}', 0.0):.4f}" for i in range(3) for j in range(4))
                    print(affine.text)

            # view_registration.insert(0, new_view_transform)

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
        # self.save_xml()