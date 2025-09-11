import zarr
import s3fs
import numpy as np
import boto3
import json

class SavePoints:
    def __init__(self, label_entries, n5_prefix):
        self.label_entries = label_entries
        self.n5_prefix = n5_prefix
        self.s3_filesystem = s3fs.S3FileSystem()
        self.default_block_size = 300000

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

    def save_interest_points_to_n5(self):
        for label_entry in self.label_entries:
            n5_path = label_entry['ip_list']['n5_path']    
            
            if self.n5_prefix.startswith("s3://"):
                output_path = self.n5_prefix + n5_path + "/interestpoints"
                store = s3fs.S3Map(root=output_path, s3=self.s3_filesystem, check=False)
                root = zarr.group(store=store, overwrite=False)
            else:
                output_path = self.n5_prefix + n5_path + "/interestpoints"
                store = zarr.N5Store(output_path)
                root = zarr.group(store, overwrite=False)

            id_dataset = "id"
            loc_dataset = "loc"

            if self.n5_prefix.startswith("s3://"):
                id_path = f"{output_path}/id"
                loc_path = f"{output_path}/loc"
                attrs_dict = dict(root.attrs)
                self.write_json_to_s3(id_path, loc_path, attrs_dict)

            interest_points = [point[1] for point in label_entry['ip_list']['interest_points']]
            interest_point_ids = np.arange(len(interest_points), dtype=np.uint64).reshape(-1, 1)
            n = 3

            if len(interest_points) > 0:
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

    def run(self):
        self.save_interest_points_to_n5()
        return 1