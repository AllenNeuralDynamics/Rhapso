

import boto3
import zarr
import s3fs
import numpy as np
import pandas as pd
from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame

class MatchProcessor:
    def __init__(self, base_path, xml_file_path):
        self.base_path = base_path
        self.xml_file_path = xml_file_path
        self.view_setup_timepoint_array = []
        self.view_match_dic = {}
        self.result = {}
        self.store = self.initialize_store()
        self.total_matches = 0
        self.s3 = boto3.client("s3")

    def initialize_store(self):
        if self.base_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            return s3fs.S3Map(root=self.base_path, s3=s3, check=False)
        else:
            return zarr.N5Store(self.base_path)
        
    def fetch_xml(self, file_path):
        if file_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            with s3.open(file_path, "r") as file:
                return file.read()
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()

    def fetch_from_s3(self, s3,  full_path):
        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=full_path, s3=s3)
            # Check if it's a valid Zarr array
        try:
            zarray = zarr.open_array(store, mode="r")
            return  zarray[:]
        except zarr.errors.ArrayNotFoundError:
            print(f"Zarr array not found at: {full_path}")
            return
   

    def load_dataframe(self):
        xml_file = self.fetch_xml(self.xml_file_path)
        processor = XMLToDataFrame(xml_file)
        dataframes = processor.run()

        df = pd.DataFrame(dataframes["image_loader"])
        self.view_setup_timepoint_array = df[["view_setup", "timepoint"]].to_numpy()

        return self.view_setup_timepoint_array

    def get_matches(self, view_id):
        setup_id, timepoint = view_id
        group_path = f"tpId_{timepoint}_viewSetupId_{setup_id}/beads"
        if self.base_path.startswith("s3://"):
            full_path = self.base_path +"/"+ group_path + "/correspondences/"
            
            try:
                s3 = s3fs.S3FileSystem(anon=False)
                store = s3fs.S3Map(root=full_path, s3=s3)
                root = zarr.open(store, mode='r')
                id_map = root.attrs['idMap']
                matches_array = root["data"][:]
                print(f"Retrieved {len(matches_array)} matches for view {view_id}")
                self.total_matches += len(matches_array)
                self.view_match_dic[(setup_id, timepoint)] = matches_array
                
                return matches_array, id_map
            except:
                print(f"no correspondences found in {full_path}")
                return
        else:
            try:
                root = zarr.group(store=self.store, overwrite=False)
                group = root[group_path]
                correspondences = group["correspondences"]

                if "data" not in correspondences:
                    print(f"No match data found for view {view_id}")
                    return np.array([], dtype=np.uint64), {}

                matches_array = correspondences["data"][:]
                id_map = correspondences.attrs.get("idMap", {})

                print(f"Retrieved {len(matches_array)} matches for view {view_id}")
                self.total_matches += len(matches_array)
                self.view_match_dic[(setup_id, timepoint)] = matches_array

                return matches_array, id_map
            except KeyError:
                print(f"No data found for view {view_id}")
                return np.array([], dtype=np.uint64), {}

    def get_ips(self, view_id):
        if self.base_path.startswith("s3://"):
            setup_id, timepoint = view_id[0], view_id[1]
            group_path = f"tpId_{timepoint}_viewSetupId_{setup_id}/beads/interestpoints/loc"
            s3 = s3fs.S3FileSystem(anon=False)
            full_path = self.base_path +"/"+group_path
            store = s3fs.S3Map(root=full_path, s3=s3)
            # Check if it's a valid Zarr array
            try:
                zarray = zarr.open_array(store, mode="r")
                data = zarray[:]
                return data
            except zarr.errors.ArrayNotFoundError:
                print(f"Zarr array not found at: {group_path}")
                return
        else:
            try:
                setup_id, timepoint = view_id[0], view_id[1]
                group_path = f"tpId_{timepoint}_viewSetupId_{setup_id}/beads/"
                
                root = zarr.group(store=self.store, overwrite=False)
                group = root[group_path]
                ips = group["interestpoints"]["loc"]
                ips_array = ips[:]

                return ips_array
            except KeyError:
                print(f"No data found for view {view_id}")
                return np.array([], dtype=np.uint64), {}

    def match_ips_pairs(self, matches, ips, view_id, id_map):
        setup_id, timepoint = view_id[0], view_id[1]
        ips_result = []
        for arr in matches:
            if len(arr) == 3:
                ind_1 = arr[0]
                ind_2 = arr[1]
                bead = arr[2]

                for k, v in id_map.items():
                    if v == bead:
                        corr_id = k.split(",")
                        corr_id = (corr_id[1], corr_id[0])
                        corr_data = self.get_ips(corr_id)
                        match = {"p1": ips[ind_1], "p2": corr_data[ind_2], "bead": bead}
                ips_result.append(match)
        self.result[(timepoint, setup_id)] = ips_result

    def collect_matches(self, view_ids):
        matches = {}
        for view_id in view_ids:
            try:
                matches, id_map = self.get_matches(list(view_id))
                ips = self.get_ips(view_id)
                self.match_ips_pairs(matches, ips, view_id, id_map)
            except:
                continue
        return matches, ips, view_id

    def run(self):
        view_setup_timepoint_array = self.load_dataframe()
        self.collect_matches(view_setup_timepoint_array)
        print(self.result, self.total_matches)
        return self.result, self.total_matches

