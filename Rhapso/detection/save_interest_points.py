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
    
    def write_one_block_dataset(self, root, name, data, dtype, attrs):
        """
        Write a dataset as exactly one block/chunk.
        Rewrites metadata and overwrites chunk 0 without deleting.
        """
        data = np.asarray(data, dtype=dtype)

        # Empty datasets can have shape 0, but chunk dims cannot be 0.
        chunks = tuple(max(1, dim) for dim in data.shape)

        if name in root:
            arr = zarr.creation.create(
                shape=data.shape,
                chunks=chunks,
                dtype=dtype,
                compressor=zarr.GZip(),
                store=root.store,
                path=f"{root.path}/{name}" if root.path else name,
                overwrite=True,
            )

            if data.size > 0:
                arr[...] = data

        else:
            arr = root.create_dataset(
                name,
                data=data,
                dtype=dtype,
                chunks=chunks,
                compressor=zarr.GZip(),
            )

        for k, v in attrs.items():
            arr.attrs[k] = v

        return arr
    
    def save_intensities_to_n5(self, view_id, n5_path):
        """
        Write intensities into an N5 group.
        """
        if self.n5_output_file_prefix.startswith("s3://"):
            output_path = self.n5_output_file_prefix + n5_path + "/interestpoints"
            store = s3fs.S3Map(root=output_path, s3=self.s3_filesystem, check=False)
            root = zarr.group(store=store, overwrite=False)
            root.attrs["n5"] = "4.0.0"
        
        else:
            store = zarr.N5Store(self.n5_output_file_prefix + n5_path + "/interestpoints")
            root = zarr.group(store, overwrite=False)
            root.attrs["n5"] = "4.0.0"

        intensities_path = "intensities"

        try:
            points_for_view = self.consolidated_data.get(view_id, [])

            if len(points_for_view) > 0:
                intensities = np.asarray(
                    [point[1] for point in points_for_view],
                    dtype=np.float32,
                )
            else:
                intensities = np.empty((0,), dtype=np.float32)

            num_intensities = intensities.shape[0]

            self.write_one_block_dataset(
                root=root,
                name=intensities_path,
                data=intensities,
                dtype="f4",
                attrs={
                    "dimensions": [num_intensities],
                    "blockSize": [max(num_intensities, 1)],
                },
            )

        except Exception as e:
            print(f"Error writing intensities dataset at {intensities_path}: {e}")
            raise
    
    def save_interest_points_to_n5(self, view_id, n5_path): 
        """
        Write interest point IDs and 3D locations into an N5 group.
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

        points_for_view = self.consolidated_data.get(view_id, [])

        if len(points_for_view) > 0:
            interest_points = np.asarray(
                [point[0] for point in points_for_view],
                dtype=np.float64,
            ).reshape(-1, 3)

            num_points = interest_points.shape[0]

            interest_point_ids = np.arange(
                num_points,
                dtype=np.uint64,
            ).reshape(-1, 1)

        else:
            interest_points = np.empty((0, 3), dtype=np.float64)
            interest_point_ids = np.empty((0, 1), dtype=np.uint64)
            num_points = 0

        self.write_one_block_dataset(
            root=root,
            name=id_dataset,
            data=interest_point_ids,
            dtype="u8",
            attrs={
                "dimensions": [num_points, 1],
                "blockSize": [max(num_points, 1), 1],
            },
        )

        self.write_one_block_dataset(
            root=root,
            name=loc_dataset,
            data=interest_points,
            dtype="f8",
            attrs={
                "dimensions": [num_points, 3],
                "blockSize": [max(num_points, 1), 3],
            },
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