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

    def write_json_to_s3(self, id_dataset_path, loc_dataset_path, attributes):
        """
        Write attributes file into both the ID and LOC dataset directories on S3.
        """
        bucket, key = id_dataset_path.replace("s3://", "", 1).split("/", 1)
        json_path = key + "/attributes.json"
        json_bytes = json.dumps(attributes).encode("utf-8")
        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=json_path, Body=json_bytes)

        bucket, key = loc_dataset_path.replace("s3://", "", 1).split("/", 1)
        json_path = key + "/attributes.json"
        json_bytes = json.dumps(attributes).encode("utf-8")
        s3.put_object(Bucket=bucket, Key=json_path, Body=json_bytes)

    def write_one_block_dataset(self, root, name, data, dtype, attrs):
        """
        Write a points as one block/chunk.
        """
        data = np.asarray(data, dtype=dtype)
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
                name=name,
                data=data,
                dtype=dtype,
                chunks=chunks,
                compressor=zarr.GZip(),
            )

        for k, v in attrs.items():
            arr.attrs[k] = v

        return arr

    def save_interest_points_to_n5(self):
        for label_entry in self.label_entries:
            n5_path = label_entry["ip_list"]["n5_path"]

            if self.n5_prefix.startswith("s3://"):
                output_path = self.n5_prefix + n5_path + "/interestpoints"
                store = s3fs.S3Map(root=output_path, s3=self.s3_filesystem, check=False)
                root = zarr.group(store=store, overwrite=False)
            else:
                output_path = self.n5_prefix + n5_path + "/interestpoints"
                store = zarr.N5Store(output_path)
                root = zarr.group(store, overwrite=False)

            root.attrs["pointcloud"] = "1.0.0"
            root.attrs["type"] = "list"
            root.attrs["list version"] = "1.0.0"

            id_dataset = "id"
            loc_dataset = "loc"

            if self.n5_prefix.startswith("s3://"):
                id_path = f"{output_path}/id"
                loc_path = f"{output_path}/loc"
                attrs_dict = dict(root.attrs)
                self.write_json_to_s3(id_path, loc_path, attrs_dict)

            interest_points = np.asarray(
                [point[1] for point in label_entry["ip_list"]["interest_points"]],
                dtype=np.float64,
            ).reshape(-1, 3)

            num_points = interest_points.shape[0]

            interest_point_ids = np.arange(
                num_points,
                dtype=np.uint64,
            ).reshape(-1, 1)

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

    def run(self):
        self.save_interest_points_to_n5()