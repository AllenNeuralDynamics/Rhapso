import numpy as np
import copy

"""
Model and Tile Setup initializes models, tiles and view pair matches
"""

class ModelAndTileSetup():
    def __init__(self, data_map_list):
        self.data_map_list = data_map_list
        self.view_transform_matrices = None
        self.corresponding_interest_points = None
        self.connected_views = None
        self.interest_points = None
        self.view_id_set = None
        self.label_map = None
        self.pairs = []
        self.tiles = {}
    
    def apply_transform(self, point, matrix):
        """
        Applies a 3D affine transformation matrix to a point using homogeneous coordinates.
        """
        point_homogeneous = np.append(point, 1)
        transformed_point = matrix.dot(point_homogeneous)[:3]  
        return transformed_point
    
    def setup_point_matches_from_interest_points(self):
        """
        Generates transformed interest point pairs between views for downstream matching.
        """
        for data_map in self.data_map_list:
            self.corresponding_interest_points = data_map['corresponding_interest_points']
            self.connected_views = data_map['connected_views']
            self.interest_points = data_map['interest_points']
            self.view_id_set = data_map['view_id_set']
            self.label_map = data_map['label_map_global']
            self.view_transform_matrices = data_map['view_transform_matrices']
            view_id_list = list(self.view_id_set)

            # Iterate and compare all viewIDs
            for i in range(len(view_id_list)):
                for j in range(i + 1, len(view_id_list)): 
                    
                    # Get transform matrices for view_id A and B
                    key_i = f"timepoint: {view_id_list[i][0]}, setup: {view_id_list[i][1]}"
                    key_j = f"timepoint: {view_id_list[j][0]}, setup: {view_id_list[j][1]}"

                    mA = self.view_transform_matrices.get(key_i, None)
                    mB = self.view_transform_matrices.get(key_j, None)   
                    
                    if mA is None or mB is None: continue

                    for label_a in self.label_map[key_i]:
                        for label_b in self.label_map[key_j]:

                            cp_a = [it for it in self.corresponding_interest_points.get(key_i, []) if it.get('label') == label_a]

                            ip_list_a = self.interest_points.get(key_i, {}).get(label_a, [])
                            ip_list_b = self.interest_points.get(key_j, {}).get(label_b, [])

                            inliers = []
                            for p in cp_a:
                                
                                # verify corresponding point is in ip_list_b
                                if label_a == label_b and p['corresponding_view_id'] == key_j:

                                    ip_a = ip_list_a[p['detection_id']]
                                    ip_b = ip_list_b[p['corresponding_detection_id']]

                                    interest_point_a = {
                                        'l': copy.deepcopy(ip_a),  
                                        'w': copy.deepcopy(ip_a),
                                        'index': p['detection_id']
                                    }
                                    interest_point_b = {
                                        'l': copy.deepcopy(ip_b),
                                        'w': copy.deepcopy(ip_b),
                                        'index': p['corresponding_detection_id']
                                    }

                                    transformed_l_a = self.apply_transform(interest_point_a['l'], mA)
                                    transformed_w_a = self.apply_transform(interest_point_a['w'], mA)
                                    transformed_l_b = self.apply_transform(interest_point_b['l'], mB)
                                    transformed_w_b = self.apply_transform(interest_point_b['w'], mB)

                                    interest_point_a['l'] = transformed_l_a
                                    interest_point_a['w'] = transformed_w_a
                                    interest_point_b['l'] = transformed_l_b
                                    interest_point_b['w'] = transformed_w_b

                                    interest_point_a['weight'] = 1
                                    interest_point_a['strength'] = 1
                                    interest_point_b['weight'] = 1
                                    interest_point_b['strength'] = 1

                                    inliers.append({
                                        'p1': interest_point_a,
                                        'p2': interest_point_b,
                                        'weight': 1,
                                        'strength': 1
                                    })
                                
                            if inliers:
                                # existing = next((p for p in self.pairs if p['view'] == (key_i, key_j)), None)
                                # if existing:
                                #     existing['inliers'].extend(inliers)                               
                                # else:

                                channel_idx = sum(1 for p in self.pairs if p['view'] == (key_i, key_j))
                                self.pairs.append({
                                    'view': (key_i, key_j),
                                    'channel_idx': channel_idx,
                                    'inliers': inliers,
                                    'flipped': None 
                                })
        
        max_ch = max(p['channel_idx'] for p in self.pairs) if self.pairs else -1

        if max_ch > 0:
            self.pairs = [[p for p in self.pairs if p['channel_idx'] == ch] for ch in range(max_ch + 1)]
    
    def run(self):
        """
        Executes the entry point of the script.
        """
        self.setup_point_matches_from_interest_points()

        return self.pairs, self.view_id_set