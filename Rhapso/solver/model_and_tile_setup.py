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
    
    def create_default_model_3d(self):
        """
        Returns a default 3D rigid transformation model with identity rotation and zero translation.
        """
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
        """
        Initializes default transformation models and parameters for affine and rigid alignment.
        """
        self.model = {
            'a' : self.create_default_model_3d(),
            'b' : self.create_default_model_3d(),
            'regularized': None,
            'cost' : 1.7976931348623157e+308,
            'l1' : 0.900000,
            'lambda' : 0.100000
        }
    
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
        view_id_list = list(self.view_id_set)

        # Iterate and compare all viewIDs
        for i in range(len(view_id_list)):
            for j in range(i + 1, len(view_id_list)): 
                
                # Get transform matrices for view_id A and B
                key_i = f"timepoint: {view_id_list[i][0]}, setup: {view_id_list[i][1]}"
                key_j = f"timepoint: {view_id_list[j][0]}, setup: {view_id_list[j][1]}"

                if key_i == 'timepoint: 0, setup: 0' and key_j == 'timepoint: 0, setup: 1':
                    print("start")

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
                    self.pairs.append({
                        'view': (key_i, key_j),
                        'inliers': inliers,
                        'flipped': None 
                    })
    
    def run(self):
        """
        Executes the entry point of the script.
        """
        self.setup_point_matches_from_interest_points()
        self.create_models()

        return self.model, self.pairs