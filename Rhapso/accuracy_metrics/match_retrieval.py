import boto3
import zarr
import s3fs
import numpy as np
import pandas as pd
from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame


class MatchProcessor:
    def __init__(self, base_path, xml_file_path, file_source, xml_bucket_name):
        self.base_path = base_path
        self.xml_file_path = xml_file_path
        self.file_source = file_source
        self.view_setup_timepoint_array = []
        self.view_match_dic = {}
        self.result = {}
        self.store = self.initialize_store()
        self.total_matches = 0
        self.s3 = boto3.client("s3")
        self.xml_bucket_name = xml_bucket_name

    def initialize_store(self):
        if self.base_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            return s3fs.S3Map(root=self.base_path, s3=s3, check=False)
        else:
            return zarr.N5Store(self.base_path)

    def fetch_from_s3(self, s3, bucket_name, input_file):
        response = s3.get_object(Bucket=bucket_name, Key=input_file)
        return response["Body"].read().decode("utf-8")

    def fetch_local_xml(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def load_dataframe(self, file_source):

        if self.file_source == "s3":
            xml_file = self.fetch_from_s3(
                self.s3, self.xml_bucket_name, self.xml_file_path
            )
        elif self.file_source == "local":
            xml_file = self.fetch_local_xml(self.xml_file_path)

        processor = XMLToDataFrame(xml_file, "metrics")
        dataframes = processor.run()

        df = pd.DataFrame(dataframes["image_loader"])
        self.view_setup_timepoint_array = df[["view_setup", "timepoint"]].to_numpy()

        return self.view_setup_timepoint_array

    def get_matches(self, store, view_id):
        setup_id, timepoint = view_id
        group_path = f"tpId_{timepoint}_viewSetupId_{setup_id}/beads"

        try:
            root = zarr.group(store=store, overwrite=False)
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
        setup_id, timepoint = view_id[0], view_id[1]
        group_path = f"tpId_{timepoint}_viewSetupId_{setup_id}/beads/"

        try:
            root = zarr.group(store=self.store, overwrite=False)
            group = root[group_path]
            ips = group["interestpoints"]

            if "loc" not in ips:
                print(f"No loc data found for view {view_id}")
                return np.array([], dtype=np.uint64), {}

            ips_array = ips["loc"][:]

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

    def collect_matches(self, store, view_ids):
        matches = {}
        for view_id in view_ids:
            matches, id_map = self.get_matches(store, list(view_id))
            ips = self.get_ips(view_id)
            self.match_ips_pairs(matches, ips, view_id, id_map)
        return matches, ips, view_id

    def run(self, processor):
        # processor = MatchProcessor(base_path, xml_file)
        view_setup_timepoint_array = processor.load_dataframe("local")
        processor.collect_matches(processor.store, view_setup_timepoint_array)
        return self.result, self.total_matches


# Example usage
if __name__ == "__main__":
    base_path = "/Users/ai/Downloads/IP_TIFF_XML/interestpoints.n5"
    xml_file = "/Users/ai/Downloads/IP_TIFF_XML/dataset.xml~1"
    xml_bucket_name = None

    processor = MatchProcessor(base_path, xml_file, "local", xml_bucket_name)
    processor.run(processor)
