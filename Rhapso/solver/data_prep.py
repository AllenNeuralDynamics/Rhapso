import zarr
import json
import os
import s3fs

"""
Data Prep fetches and preps n5 interest points data
"""

class DataPrep():
    def __init__(self, interest_points_df, view_transform_matrices, xml_file_path, n5_input_path):
        self.interest_points_df = interest_points_df
        self.view_transform_matrices = view_transform_matrices
        self.xml_file_path = xml_file_path
        self.n5_input_path = n5_input_path

        self.connected_views = {} 
        self.corresponding_interest_points = {}
        self.interest_points = {}
        self.label_map_global = {}
    
    def get_connected_views_from_n5(self):
        """
        Loads connected view mappings from N5 metadata, supporting both S3 and local sources.
        """
        for _, row in self.interest_points_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"

            if self.n5_input_path.startswith("s3://"):
                s3 = s3fs.S3FileSystem(anon=False)
                root = self.n5_input_path.replace("s3://", "", 1)
                path = root + "interestpoints.n5"
                store = s3fs.S3Map(root=path, s3=s3)
                root = zarr.open(store, mode='r')
                correspondences_key = f"{row['path']}/correspondences"
                try:
                    self.connected_views[view_id] = root[correspondences_key].attrs["idMap"]
                except:
                    print(f"No connected views for tile {view_id}")

            else:
                n5_root = os.path.join(self.n5_input_path, "interestpoints.n5")
                store = zarr.N5Store(n5_root)
                root = zarr.open(store, mode="r")
                correspondences_key = f"{row['path']}/correspondences"
                try:
                    self.connected_views[view_id] = self.load_json_data(correspondences_key)
                except:
                    print(f"No connected views for tile {view_id}")            
    
    def load_json_data(self, json_path):
        try:
            path = self.n5_input_path + json_path
            if not os.path.exists(path):
                return {}
            with open(path, 'r') as f:
                obj = json.load(f)
            id_map = obj.get('idMap', {})
            return id_map if isinstance(id_map, dict) else {}
        except Exception:
            return {}
        
    def get_corresponding_data_from_n5(self):
        """
        Parses and transforms corresponding interest point data from N5 format into world space coordinates.
        """
        if self.n5_input_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            root = self.n5_input_path.replace("s3://", "", 1)
            path = root + "interestpoints.n5"
            store = s3fs.S3Map(root=path, s3=s3)
        else:
            path = self.n5_input_path + "interestpoints.n5"
            store = zarr.N5Store(path)

        root = zarr.open(store, mode='r')

        for _, row in self.interest_points_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"  
            correspondences_prefix = f"{row['path']}/correspondences"
            attributes_path = 'interestpoints.n5' + f"/{row['path']}/correspondences/attributes.json"

            # Load JSON data for idMap
            if self.n5_input_path.startswith("s3://"):
                try:
                    id_map = root[correspondences_prefix].attrs['idMap']
                except Exception:
                    continue
            else:
                id_map = self.load_json_data(attributes_path)
                if not id_map:
                    continue
            
            try:
                interest_points_index_map = root[correspondences_prefix + '/data'][:]
            except (KeyError, FileNotFoundError, AttributeError, TypeError):
                print(f"⚠️ Skipping {view_id}: missing correspondences.")
                continue

            # Load corresponding interest points data
            for ip_index, corr_index, corr_group_id in interest_points_index_map:
                if corr_group_id == view_id:
                    continue

                corresponding_view_id = next((k for k, v in id_map.items() if v == int(corr_group_id)), None)
                parts = corresponding_view_id.split(',')
                timepoint, setup, label = parts[0], parts[1], parts[2]
                corresponding_view_id = f"timepoint: {timepoint}, setup: {setup}"

                ip = self.interest_points[view_id][label][int(ip_index)]
                corr_ip = self.interest_points[corresponding_view_id][label][int(corr_index)]

                if view_id not in self.corresponding_interest_points:
                    self.corresponding_interest_points[view_id] = [] 
                
                self.corresponding_interest_points[view_id].append({
                    "detection_id": ip_index,
                    "detection_p1": ip,
                    "corresponding_detection_id":  corr_index,
                    "corresponding_detection_p2": corr_ip,
                    "corresponding_view_id": corresponding_view_id,
                    "label": label
                })
    
    def get_all_interest_points_from_n5(self):
        """
        Loads raw interest point coordinates from N5 storage into memory, keyed by view ID.
        """
        if self.n5_input_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            root = self.n5_input_path.replace("s3://", "", 1)
            path = root + "interestpoints.n5"
            store = s3fs.S3Map(root=path, s3=s3)
        else:
            path = self.n5_input_path + "interestpoints.n5"
            store = zarr.N5Store(path)

        root = zarr.open(store, mode='r')

        for _, row in self.interest_points_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"  
            interestpoints_prefix = f"{row['path']}/interestpoints/loc/"
            interest_points = root[interestpoints_prefix][:]
            # interest_points = root[interestpoints_prefix][:] if interestpoints_prefix in root else np.empty((0, 3), dtype=np.float32)
            label = str(row['path']).replace('\\','/').lstrip('/').split('/', 2)[1]
            self.interest_points.setdefault(view_id, {})[label] = interest_points
                            
    def build_label_map(self):
        """
        Constructs a mapping of labels for each view ID from the interest points dataframe.
        """
        for _, row in self.interest_points_df.iterrows():
            view_id_key = f"timepoint: {row['timepoint']}, setup: {row['setup']}"
            
            if view_id_key not in self.label_map_global:
                self.label_map_global[view_id_key] = {}

            self.label_map_global[view_id_key][row['label']] = 1.0  
    
    def run(self):
        """
        Executes the entry point of the script.
        """
        self.build_label_map()
        self.get_all_interest_points_from_n5()
        self.get_corresponding_data_from_n5()
        self.get_connected_views_from_n5()

        view_id_set = set()
        for k in self.corresponding_interest_points.keys():
            try:
                parts = [p.strip() for p in k.split(',')]
                tp = parts[0].split(':')[-1].strip()
                su = parts[1].split(':')[-1].strip()
                view_id_set.add((str(tp), str(su)))
            except Exception:
                continue

        self.view_id_set = sorted(view_id_set, key=lambda x: (int(x[0]), int(x[1])))

        return self.connected_views, self.corresponding_interest_points, self.interest_points, self.label_map_global, self.view_id_set