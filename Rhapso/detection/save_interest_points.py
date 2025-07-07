import zarr
import numpy as np
import xml.etree.ElementTree as ET
import s3fs
import boto3
from io import BytesIO
import io
import json

class SaveInterestPoints:
    def __init__(self, dataframes, consolidated_data, xml_file_path, xml_bucket_name, output_bucket_name, output_file_path,
                 downsample_xy, downsample_z, min_intensity, max_intensity, sigma, threshold, file_source): 
        self.consolidated_data = consolidated_data
        self.image_loader_df = dataframes['image_loader']
        self.xml_file_path = xml_file_path
        self.xml_bucket_name = xml_bucket_name
        self.output_bucket_name = output_bucket_name
        self.output_file_path = output_file_path
        self.downsample_xy = downsample_xy
        self.downsample_z = downsample_z
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.sigma = sigma
        self.threshold = threshold
        self.file_source = file_source

        self.s3_filesystem = s3fs.S3FileSystem()
        self.overlappingOnly = "true"
        self.findMin = "true"
        self.findMax = "true"
        self.default_block_size = 300000
    
    def load_xml_file(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        return tree, root
    
    def fetch_from_s3(self, s3, bucket_name, input_file):
        response = s3.get_object(Bucket=bucket_name, Key=input_file)
        return response['Body'].read().decode('utf-8')
    
    def save_to_xml(self):
        if self.file_source == 'local':
            tree, root = self.load_xml_file(self.xml_file_path)
        
        elif self.file_source == 's3':
            s3 = boto3.client('s3')
            xml_string = self.fetch_from_s3(s3, self.xml_bucket_name, self.xml_file_path)
            tree = ET.parse(io.BytesIO(xml_string.encode('utf-8')))
            root = tree.getroot()
        
        else:
            print("input source not accepted.")
            return

        interest_points_section = root.find('.//ViewInterestPoints')
        
        if interest_points_section is None:
            interest_points_section = ET.SubElement(root, 'ViewInterestPoints')
            interest_points_section.text = '\n    ' 
        
        else:
            interest_points_section.clear()
            interest_points_section.text = '\n    '  

        for view_id, _ in self.consolidated_data.items():
            parts = view_id.split(',') 
            timepoint_part = parts[0].strip()  
            setup_part = parts[1].strip() 

            timepoint = int(timepoint_part.split(':')[1].strip())  
            setup = int(setup_part.split(':')[1].strip())
            label = "beads"
            params = "DOG (Spark) s={} t={} overlappingOnly={} min={} max={} downsampleXY={} downsampleZ={} minIntensity={} maxIntensity={}".format(
                self.sigma, self.threshold, self.overlappingOnly, self.findMin, self.findMax,
                self.downsample_xy, self.downsample_z, self.min_intensity, self.max_intensity)
            value = f"tpId_{timepoint}_viewSetupId_{setup}/beads"

            new_interest_point = ET.SubElement(interest_points_section, 'ViewInterestPointsFile', {
                'timepoint': str(timepoint),
                'setup': str(setup),
                'label': label,
                'params': params
            })
            new_interest_point.text = value
            new_interest_point.tail = '\n    '
        
        interest_points_section.tail = '\n  '

        if self.file_source == 'local':
            tree.write(self.output_file_path + '/dataset-detection.xml', encoding='utf-8', xml_declaration=True)
        
        elif self.file_source == 's3':
            xml_bytes = BytesIO()
            tree.write(xml_bytes, encoding='utf-8', xml_declaration=True)
            xml_bytes.seek(0)
            s3 = boto3.client('s3')
            object_name = self.output_file_path + '/dataset-detection.xml'  
            s3.upload_fileobj(xml_bytes, self.output_bucket_name, object_name)
        
        else:
            print("input source not accepted")
            return
    
    def write_json_to_s3(self, id_dataset_path, loc_dataset_path, attributes):
        json_path = id_dataset_path + '/attributes.json'
        json_bytes = json.dumps(attributes).encode('utf-8')
        s3 = boto3.client('s3')
        s3.put_object(Bucket=self.output_bucket_name, Key=json_path, Body=json_bytes)

        json_path = loc_dataset_path + '/attributes.json'
        json_bytes = json.dumps(attributes).encode('utf-8')
        s3 = boto3.client('s3')
        s3.put_object(Bucket=self.output_bucket_name, Key=json_path, Body=json_bytes)

    def save_intensities_to_n5(self, view_id, n5_path, root):
        intensities_path = n5_path + '/interestpoints/intensities'

        if intensities_path in root:
            try:
                del root[intensities_path]
            except Exception as e:
                print(f"Warning: failed to delete existing dataset at {intensities_path}: {e}")

        try: 
            if view_id in self.consolidated_data:
                intensities = [point[1] for point in self.consolidated_data[view_id]] 
                root.create_dataset(
                    intensities_path,
                    data=intensities,
                    dtype='f4',  
                    chunks=(self.default_block_size,),  
                    compressor=zarr.GZip()
                )
            
            else: 
                root.create_dataset(
                    intensities_path,
                    shape=(0,), 
                    dtype='f4', 
                    chunks=(1,),  
                    compressor=zarr.GZip()  
                )
        except Exception as e:
            print(f"Error creating intensities dataset at {intensities_path}: {e}")

    def save_interest_points_to_n5(self, view_id, n5_path, root): 
        dataset_path = n5_path + '/interestpoints'    

        if dataset_path in root:
            try:
                group = root[dataset_path]
                for key in list(group.keys()):
                    del group[key]
                del root[dataset_path]
            except Exception as e:
                print(f"Warning: failed to delete existing group at {dataset_path}: {e}")

         # Create group
        try:
            dataset = root.create_group(dataset_path)
        except zarr.errors.ContainsGroupError:
            dataset = root[dataset_path]  # fallback 

        # Set attributes
        dataset.attrs["pointcloud"] = "1.0.0"
        dataset.attrs["type"] = "list"
        dataset.attrs["list version"] = "1.0.0"

        # Create sub-datasets
        id_dataset = f"{dataset_path}/id"
        loc_dataset = f"{dataset_path}/loc"

        # Create attributes.json files if saving to s3
        if self.file_source == 's3':
            id_path = "output/" + id_dataset
            loc_path = "output/" + loc_dataset
            attrs_dict = dict(dataset.attrs)
            self.write_json_to_s3(id_path, loc_path, attrs_dict)

        # Prep interest points lists or create empty n5 directories if none
        if view_id in self.consolidated_data:
            interest_points = [point[0] for point in self.consolidated_data[view_id]]
            interest_point_ids = np.arange(len(interest_points), dtype=np.uint64).reshape(-1, 1)

            n = 3
            
            # Save the IDs (1 x N array)
            if id_dataset in root:
                del root[id_dataset]
            root.create_dataset(
                id_dataset,
                data=interest_point_ids,
                dtype='u8',
                chunks=(self.default_block_size,),  
                compressor=zarr.GZip()
            )

            # Save the locations (DIM x N array)
            if loc_dataset in root:
                del root[loc_dataset]
            root.create_dataset(
                loc_dataset,
                data=interest_points, 
                dtype='f8',
                chunks=(self.default_block_size, n),  
                compressor=zarr.GZip()
            )
        
        # save as empty lists
        else:
            if id_dataset in root:
                del root[id_dataset]
            root.create_dataset(
                id_dataset,
                shape=(0,), 
                dtype='u8',  
                chunks=(1,),  
                compressor=zarr.GZip()  
            )

            if loc_dataset in root:
                del root[loc_dataset]
            root.create_dataset(
                loc_dataset,
                shape=(0,),  
                dtype='f8',  
                chunks=(1,), 
                compressor=zarr.GZip()  
            )

    def save_points(self):
        if self.file_source == 'local':
            store = zarr.N5Store(self.output_file_path)
            root = zarr.group(store, overwrite=False)
            root.attrs['n5'] =  '4.0.0'
        
        elif self.file_source == 's3':
            output_path = self.output_bucket_name + "/" + self.output_file_path
            store = s3fs.S3Map(root=output_path, s3=self.s3_filesystem, check=False)
            root = zarr.group(store=store, overwrite=False)
            root.attrs['n5'] = '4.0.0'
   
        for _, row in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['view_setup']}"
            n5_path = f"interestpoints.n5/tpId_{row['timepoint']}_viewSetupId_{row['view_setup']}/beads"
            self.save_interest_points_to_n5(view_id, n5_path, root)
            self.save_intensities_to_n5(view_id, n5_path, root)
        self.save_to_xml()

    def run(self):
        self.save_points()