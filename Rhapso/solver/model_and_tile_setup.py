import numpy as np
import copy

# This class sets up the default models, tiles, and view pair matches

class ModelAndTileSetup():
    def __init__(self, connected_views, corresponding_interest_points, interest_points, view_transform_matrices, view_id_set, label_map):
        self.corresponding_interest_points = corresponding_interest_points
        self.view_transform_matrices = view_transform_matrices
        self.connected_views = connected_views
        self.interest_points = interest_points
        self.view_id_set = view_id_set
        self.label_map = label_map
        self.pairs = []
        self.tiles = {}

    def create_tiles(self):
        view_id_list = list(self.view_id_set)
        for view_id in view_id_list:
            view_id_key = f"timepoint: {view_id[0]}, setup: {view_id[1]}"
            matches = [
                {
                    'p1': match['detection_p1'], 
                    'p2': match['corresponding_detection_p2'],
                    'strength': 1.0,  
                    'weight': 1.0    
                }
                for match in self.corresponding_interest_points.get(view_id_key, [])
            ]
            self.tiles[view_id_key] = {
                'model': self.model,
                'matches' : matches,
                'connected_tiles': self.connected_views[view_id_key],
                'cost': 0.0,
                'distance': 0.0
            }
    
    def create_default_interpolated_affine_model_3d(self):
        return {
            "m00": 1.0, "m01": 0.0, "m02": 0.0, "m03": 0.0,
            "m10": 0.0, "m11": 1.0, "m12": 0.0, "m13": 0.0,
            "m20": 0.0, "m21": 0.0, "m22": 1.0, "m23": 0.0,
            "i00": 1.0, "i01": 0.0, "i02": 0.0, "i03": -0.0,
            "i10": 0.0, "i11": 1.0, "i12": 0.0, "i13": -0.0,
            "i20": 0.0, "i21": 0.0, "i22": 1.0, "i23": -0.0,
            "cost": 1.7976931348623157e+308,  
            "isInvertible": True
        }
    
    def create_default_rigid_model_3d(self):
        return {
            "m00": 1.0, "m01": 0.0, "m02": 0.0, "m03": 0.0,
            "m10": 0.0, "m11": 1.0, "m12": 0.0, "m13": 0.0,
            "m20": 0.0, "m21": 0.0, "m22": 1.0, "m23": 0.0,
            "i00": 1.0, "i01": 0.0, "i02": 0.0, "i03": 0.0,
            "i10": 0.0, "i11": 1.0, "i12": 0.0, "i13": 0.0,
            "i20": 0.0, "i21": 0.0, "i22": 1.0, "i23": 0.0,
            "cost": 1.7976931348623157e+308,  
            "isInvertible": True
        }

    def create_default_affine_model_3d(self):
        return {
            "m00": 1.0, "m01": 0.0, "m02": 0.0, "m03": 0.0,
            "m10": 0.0, "m11": 1.0, "m12": 0.0, "m13": 0.0,
            "m20": 0.0, "m21": 0.0, "m22": 1.0, "m23": 0.0,
            "i00": 1.0, "i01": 0.0, "i02": 0.0, "i03": 0.0,
            "i10": 0.0, "i11": 1.0, "i12": 0.0, "i13": 0.0,
            "i20": 0.0, "i21": 0.0, "i22": 1.0, "i23": 0.0,
            "cost": 1.7976931348623157e+308, 
            "isInvertible": True
        }
    
    def create_models(self):
        self.model = {
            'a' : self.create_default_affine_model_3d(),
            'affine' : self.create_default_interpolated_affine_model_3d(),
            'afs' : [1.0 if i in [0, 4, 8] else 0.0 for i in range(13)],
            'b' : self.create_default_rigid_model_3d(),
            'bfs' : [1.0 if i in [0, 4, 8] else 0.0 for i in range(13)],
            'cost' : 1.7976931348623157e+308,
            'l1' : 0.900000,
            'lambda' : 0.100000
        }
    
    def apply_transform(self, point, matrix):
        point_homogeneous = np.append(point, 1)
        transformed_point = matrix.dot(point_homogeneous)[:3]  
        return transformed_point
    
    def assign_point_matches(self):
        for pair in self.pairs:
            pass

    def setup_point_matches_from_interest_points(self):
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

                # Get corresponding interest points for view_id A
                cp_a = self.corresponding_interest_points[key_i]

                # Get interest points for view_id A and B
                ip_list_a = self.interest_points[key_i]
                ip_list_b = self.interest_points[key_j]

                inliers = []
                for p in cp_a:
                    
                    # verify corresponding point is in ip_list_b
                    if p['corresponding_view_id'] == key_j:

                        ip_a = ip_list_a[p['detection_id']]
                        ip_b = ip_list_b[p['corresponding_detection_id']]

                        interest_point_a = {
                            'l': copy.deepcopy(ip_a),  
                            'w': copy.deepcopy(ip_a)
                        }
                        interest_point_b = {
                            'l': copy.deepcopy(ip_b),
                            'w': copy.deepcopy(ip_b)
                        }

                        transformed_l_a = self.apply_transform(interest_point_a['l'], mA)
                        transformed_w_a = self.apply_transform(interest_point_a['w'], mA)
                        transformed_l_b = self.apply_transform(interest_point_b['l'], mB)
                        transformed_w_b = self.apply_transform(interest_point_b['w'], mB)

                        interest_point_a['l'] = transformed_l_a
                        interest_point_a['w'] = transformed_w_a
                        interest_point_b['l'] = transformed_l_b
                        interest_point_b['w'] = transformed_w_b

                        inliers.append((interest_point_a, interest_point_b))
                    
                if inliers:
                    self.pairs.append(((key_i, key_j), inliers))
    
    def run(self):
        self.setup_point_matches_from_interest_points()
        self.create_models()
        self.create_tiles()
        # self.assign_point_matches()

        return self.tiles, self.model, self.pairs