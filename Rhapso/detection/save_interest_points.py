import zarr
import numpy as np
import xml.etree.ElementTree as ET
import s3fs
import boto3
from io import BytesIO
import io
import json

"""
Save Interest Points saves interest points as N5 and updates the xml with pathways
"""

class SaveInterestPoints:
    def __init__(self, dataframes, consolidated_data, xml_file_path, xml_output_file_path, n5_output_file_prefix, downsample_xy, downsample_z, min_intensity, 
                 max_intensity, sigma, threshold): 
        self.consolidated_data = consolidated_data
        self.image_loader_df = dataframes['image_loader']
        self.xml_file_path = xml_file_path
        self.xml_output_file_path = xml_output_file_path
        self.n5_output_file_prefix = n5_output_file_prefix
        self.downsample_xy = downsample_xy
        self.downsample_z = downsample_z
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.sigma = sigma
        self.threshold = threshold
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
        """
        Rebuild the <ViewInterestPoints> section and write the updated XML back
        """
        if self.xml_file_path.startswith("s3://"):
            bucket, key = self.xml_file_path.replace("s3://", "", 1).split("/", 1)
            s3 = boto3.client('s3')
            xml_string = self.fetch_from_s3(s3, bucket, key)
            tree = ET.parse(io.BytesIO(xml_string.encode('utf-8')))
            root = tree.getroot()
        else:
            tree, root = self.load_xml_file(self.xml_file_path)

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

        if self.xml_output_file_path.startswith("s3://"):
            bucket, key = self.xml_output_file_path.replace("s3://", "", 1).split("/", 1)
            xml_bytes = BytesIO()
            tree.write(xml_bytes, encoding='utf-8', xml_declaration=True)
            xml_bytes.seek(0)
            s3 = boto3.client('s3') 
            s3.upload_fileobj(xml_bytes, bucket, key)

        else:
            tree.write(self.xml_output_file_path, encoding='utf-8', xml_declaration=True)
        
    def write_json_to_s3(self, id_dataset_path, loc_dataset_path, attributes):
        """
        Write attributes file into both the ID and LOC dataset directories on S3
        """
        bucket, key = id_dataset_path.replace("s3://", "", 1).split("/", 1)
        json_path = key + '/attributes.json'
        json_bytes = json.dumps(attributes).encode('utf-8')
        s3 = boto3.client('s3')
        s3.put_object(Bucket=bucket, Key=json_path, Body=json_bytes)

        bucket, key = loc_dataset_path.replace("s3://", "", 1).split("/", 1)
        json_path = key + '/attributes.json'
        json_bytes = json.dumps(attributes).encode('utf-8')
        s3 = boto3.client('s3')
        s3.put_object(Bucket=bucket, Key=json_path, Body=json_bytes)

    def save_intensities_to_n5(self, view_id, n5_path):
        """
        Write intensities into an N5 group
        """
        if self.n5_output_file_prefix.startswith("s3://"):
            output_path = self.n5_output_file_prefix + n5_path + "/interestpoints"
            store = s3fs.S3Map(root=output_path, s3=self.s3_filesystem, check=False)
            root = zarr.group(store=store, overwrite=False)
            root.attrs['n5'] = '4.0.0'
        
        else:
            store = zarr.N5Store(self.n5_output_file_prefix + n5_path + "/interestpoints")
            root = zarr.group(store, overwrite=False)
            root.attrs['n5'] =  '4.0.0'
        
        intensities_path = 'intensities'

        if intensities_path in root:
            try:
                del root[intensities_path]
            except Exception as e:
                print(f"Warning: failed to delete existing dataset at {intensities_path}: {e}")

        try: 
            if view_id in self.consolidated_data:
                intensities = [point[1] for point in self.consolidated_data[view_id]] 
                dataset = root.create_dataset(
                    intensities_path,
                    data=intensities,
                    dtype='f4',  
                    chunks=(self.default_block_size,),  
                    compressor=zarr.GZip()
                )
                dataset.attrs["dimensions"] = [1, len(intensities)]
                dataset.attrs["blockSize"] = [1, self.default_block_size]
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

    def save_interest_points_to_n5(self, view_id, n5_path): 
        """
        Write interest point IDs and 3D locations into an N5 group
        """
        if self.n5_output_file_prefix.startswith("s3://"):
            output_path = self.n5_output_file_prefix + n5_path + "/interestpoints"
            store = s3fs.S3Map(root=output_path, s3=self.s3_filesystem, check=False)
            root = zarr.group(store=store, overwrite=False)
            root.attrs["pointcloud"] = "1.0.0"
            root.attrs["type"] = "list"
            root.attrs["list version"] = "1.0.0"

        else:
            store = zarr.N5Store(self.n5_output_file_prefix + n5_path + "/interestpoints")
            root = zarr.group(store, overwrite=False)
            root.attrs["pointcloud"] = "1.0.0"
            root.attrs["type"] = "list"
            root.attrs["list version"] = "1.0.0"

        id_dataset = "id"
        loc_dataset = "loc"

        if self.n5_output_file_prefix.startswith("s3://"):
            id_path = f"{output_path}/id"
            loc_path = f"{output_path}/loc"
            attrs_dict = dict(root.attrs)
            self.write_json_to_s3(id_path, loc_path, attrs_dict)

        if (view_id in self.consolidated_data) and (len(self.consolidated_data[view_id]) > 0):
            interest_points = [point[0] for point in self.consolidated_data[view_id]]
            interest_point_ids = np.arange(len(interest_points), dtype=np.uint64).reshape(-1, 1)
            n = 3

            if id_dataset in root:
                del root[id_dataset]
            root.create_dataset(
                id_dataset,
                data=interest_point_ids,
                dtype='u8',
                chunks=(self.default_block_size,),
                compressor=zarr.GZip()
            )

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
        """
        Orchestrate interest points and intensities into an N5 layout - inject attributes file
        """
        for _, row in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['view_setup']}"
            n5_path = f"interestpoints.n5/tpId_{row['timepoint']}_viewSetupId_{row['view_setup']}/beads"
            self.save_interest_points_to_n5(view_id, n5_path)
            self.save_intensities_to_n5(view_id, n5_path)

        path = self.n5_output_file_prefix + "interestpoints.n5"
        
        if path.startswith("s3://"):
            bucket_key = path.replace("s3://", "", 1)
            store = s3fs.S3Map(root=bucket_key, s3=self.s3_filesystem, check=False)
            root = zarr.group(store=store, overwrite=False)
            root.attrs['n5'] = '4.0.0'
        else:
            store = zarr.N5Store(path)
            root = zarr.group(store, overwrite=False)
            root.attrs['n5'] =  '4.0.0'

    def run(self):
        """
        Executes the entry point of the script.
        """
        self.save_points()
        self.save_to_xml()