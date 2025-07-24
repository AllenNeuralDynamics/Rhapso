import zarr
import json
import os
import numpy as np
import s3fs

# This class fetches and preps data from n5

class DataPrep():
    def __init__(self, interest_points_df, view_transform_matrices, fixed_views, data_prefix, file_source):
        self.interest_points_df = interest_points_df
        self.view_transform_matrices = view_transform_matrices
        self.fixed_views = fixed_views
        self.data_prefix = data_prefix
        self.file_source = file_source
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

            if self.file_source == 's3':
                s3 = s3fs.S3FileSystem(anon=False)
                store = s3fs.S3Map(root=self.data_prefix, s3=s3)
                root = zarr.open(store, mode='r')
                correspondences_key = f"{row['path']}/correspondences"

                self.connected_views[view_id] = root[correspondences_key].attrs['idMap']

            elif self.file_source == 'local':
                correspondences_key = f"{row['path']}/correspondences/attributes.json"
                full_path = os.path.join(self.data_prefix, correspondences_key)

                if os.path.exists(full_path):
                    with open(full_path, 'r') as file:
                        data = json.load(file)
                        self.connected_views[view_id] = data.get('idMap', {})
                else:
                    print(f"Attributes file not found for view {view_id}: {full_path}")
    
    def load_json_data(self, json_path):
        """
        Loads JSON data from either an S3 path or local filesystem.
        """
        if json_path.startswith('s3://'):
            fs = s3fs.S3FileSystem(anon=False)  
            with fs.open(json_path, 'r') as file:
                data = json.load(file)
        else:
            with open(json_path, 'r') as file:
                data = json.load(file)
        
        return data
    
    def transform_points(self, view_id, corresponding_view_id, ip, corr_ip):
        """
        Transforms two points from local to world coordinates using their view transform matrices.
        """
        transform_matrix = self.view_transform_matrices[view_id]
        corresponding_transform_matrix = self.view_transform_matrices[corresponding_view_id]

        # Homogeneous coordinates
        ip_hom = np.append(np.array(ip), 1)
        corr_ip_hom = np.append(np.array(corr_ip), 1)

        # Transform both points into world space
        ip_world = transform_matrix @ ip_hom
        corr_ip_world = corresponding_transform_matrix @ corr_ip_hom

        ip_world = ip_world[:-1] 
        corr_ip_world = corr_ip_world[:-1]  

        return ip_world, corr_ip_world
        
    def get_corresponding_data_from_n5(self):
        """
        Parses and transforms corresponding interest point data from N5 format into world space coordinates.
        """
        if self.file_source == 's3':
            s3 = s3fs.S3FileSystem(anon=False)
            store = s3fs.S3Map(root=self.data_prefix, s3=s3)
        elif self.file_source == 'local':
            store = zarr.N5Store(self.data_prefix)

        root = zarr.open(store, mode='r')

        for _, row in self.interest_points_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"  
            correspondences_prefix = f"{row['path']}/correspondences"
            attributes_path = self.data_prefix + f"/{row['path']}/correspondences/attributes.json"

            # Load JSON data for idMap
            if self.file_source == 's3':
                id_map = root[correspondences_prefix].attrs['idMap']
            elif self.file_source == 'local':
                id_map = self.load_json_data(attributes_path)['idMap']
                
            interest_points_index_map = root[correspondences_prefix + '/data'][:]

            # Load corresponding interest points data
            for ip_index, corr_index, corr_group_id in interest_points_index_map:
                if corr_group_id == view_id:
                    continue
                corresponding_view_id = next((k for k, v in id_map.items() if v == int(corr_group_id)), None)
                parts = corresponding_view_id.split(',')
                timepoint, setup, label = parts[0], parts[1], parts[2]
                corresponding_view_id = f"timepoint: {timepoint}, setup: {setup}"

                # ip, corr_ip = self.transform_points(view_id, corresponding_view_id, self.interest_points[view_id][int(ip_index)], self.interest_points[corresponding_view_id][int(corr_index)])

                ip = self.interest_points[view_id][int(ip_index)]
                corr_ip = self.interest_points[corresponding_view_id][int(corr_index)]

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
        if self.file_source == 's3':
            s3 = s3fs.S3FileSystem(anon=False)
            store = s3fs.S3Map(root=self.data_prefix, s3=s3)
        elif self.file_source == 'local':
            store = zarr.N5Store(self.data_prefix)

        root = zarr.open(store, mode='r')

        for _, row in self.interest_points_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"  
            interestpoints_prefix = f"{row['path']}/interestpoints/loc/"
            interest_points = root[interestpoints_prefix][:]
            
            if view_id in self.interest_points:
                print("duplicated viewID, skipping.")
                continue
            else:
                self.interest_points[view_id] = []
                self.interest_points[view_id] = interest_points
                            
    def build_label_map(self):
        """
        Constructs a mapping of labels for each view ID from the interest points dataframe.
        """
        for _, row in self.interest_points_df.iterrows():
            view_id_key = f"timepoint: {row['timepoint']}, setup: {row['setup']}"
            
            if view_id_key not in self.label_map_global:
                self.label_map_global[view_id_key] = {}

            self.label_map_global[view_id_key][row['label']] = 1.0
    
    def create_view_id_set(self):
        """
        Generates a set of unique (timepoint, setup) view ID pairs from the dataframe.
        """
        self.view_id_set = set(zip(self.interest_points_df['timepoint'], self.interest_points_df['setup']))
    
    def run(self):
        """
        Executes the entry point of the script.
        """
        self.create_view_id_set()
        self.build_label_map()
        self.get_all_interest_points_from_n5()
        self.get_corresponding_data_from_n5()
        self.get_connected_views_from_n5()

        return self.connected_views, self.corresponding_interest_points, self.interest_points, self.label_map_global, self.view_id_set