import os
import json
import numpy as np
import pandas as pd
import s3fs
import zarr
from Rhapso.evaluation.save_metrics import JSONFileHandler
from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame


class DetectionOutput:
    def __init__(self, base_path, xml_file_path, output_path):
        self.base_path = base_path
        self.xml_file_path = xml_file_path
        self.output_path = output_path
        self.count_ips = 0

    def initialize_store(self, path):
        if path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            return s3fs.S3Map(root=path, s3=s3, check=False)
        else:
            return zarr.N5Store(path)

    def fetch_xml(self, file_path):
        if file_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            with s3.open(file_path, "r") as file:
                return file.read()
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()

    def load_dataframe(self):
        xml_file = self.fetch_xml(self.xml_file_path)
        processor = XMLToDataFrame(xml_file)
        dataframes = processor.run()
        df = pd.DataFrame(dataframes["image_loader"])
        return df[["view_setup", "timepoint"]].to_numpy()

    def read_detection_output(self, full_path):
        if full_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            store = s3fs.S3Map(root=full_path, s3=s3)
            # Check if it's a valid Zarr array
            try:
                zarray = zarr.open_array(store, mode="r")
                data = zarray[:]
            except zarr.errors.ArrayNotFoundError:
                print(f"Zarr array not found at: {full_path}")
                return
        else:
            full_path = full_path.rstrip("/")
            components = full_path.split("/")
            try:
                n5_index = next(i for i, c in enumerate(components) if c.endswith(".n5"))
            except StopIteration:
                raise ValueError("No .n5 directory found in path")

            dataset_path = "/".join(components[: n5_index + 1])
            dataset_rel_path = "/".join(components[n5_index + 1 :])
            store = zarr.N5Store(dataset_path)
            root = zarr.open(store, mode="r")

            if dataset_rel_path not in root:
                print(f"Skipping: {dataset_rel_path} (not found)")
                return

            zarray = root[dataset_rel_path]
            data = zarray[:]

        self.count_ips += len(data)

    def run(self):
        self.view_setup_timepoint_array = self.load_dataframe()
        for view_id in self.view_setup_timepoint_array:
            view = view_id.tolist()
            path = f"{self.base_path}/tpId_{view[1]}_viewSetupId_{view[0]}/beads/interestpoints/loc/"
            self.read_detection_output(path)

        saveJSON = JSONFileHandler(self.output_path)
        saveJSON.update("Total IPS", self.count_ips)

