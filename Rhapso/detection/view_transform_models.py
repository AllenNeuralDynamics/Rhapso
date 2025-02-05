import pandas as pd
import numpy as np

# This component creates transform matrices for each viewID matrice using xmlDF -> ViewRegistrations/ViewTransform/Affine-matrices

# TODO - this component needs zarr functionality, zarr xml view transforms elements only have 1 
# matrice per viewID, tiff has 2 - we should be able to skip the dot concatenate step

class ViewTransformModels:

    def __init__(self, df):
        self.view_registrations_df = df.get('view_registrations', pd.DataFrame())
        self.calibration_matrices = {}
        self.rotation_matrices = {}
        self.concatenated_matrices = {}

    def create_transform_matrices(self):
        # parse DF for view_transform matrices
        for _, row in self.view_registrations_df.iterrows():       
            if row['type'] == 'affine':      
                
                # create affine matrix
                affine_values = np.fromstring(row['affine'], sep=',').astype(np.float64)
                if len(affine_values) == 12:
                    affine_values = np.append(affine_values, [0, 0, 0, 1])  # append homogeneous coordinates
                affine_matrix = affine_values.reshape(4, 4)

                # append matrix by row name
                view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"
                if 'calibration' in row['name'].lower():
                    self.calibration_matrices[view_id] = {
                        'affine_matrix': affine_matrix
                    }
                else:
                    self.rotation_matrices[view_id] = {
                        'affine_matrix': affine_matrix
                    }
        
            # TODO - if row['type'] == 'global':
    
    def concatenate_matrices_by_view_id(self):
        # get DOT product of rotation and calibration matrices
        for key in self.calibration_matrices:
            if key in self.rotation_matrices:
                calibration_matrix = self.calibration_matrices[key]['affine_matrix']
                rotation_matrix = self.rotation_matrices[key]['affine_matrix']
                concatenated_matrix = np.dot(rotation_matrix, calibration_matrix)
                self.concatenated_matrices[key] = concatenated_matrix
    
    def print_matrices_as_affine(self):
        # view friendly print function
        for key, matrix in self.concatenated_matrices.items():
            affine_format = "3d-affine: (" + ", ".join(f"{item:.6f}" for row in matrix for item in row) + ")"
            print(f"{key}: {affine_format}")

    def run(self):
        self.create_transform_matrices()
        self.concatenate_matrices_by_view_id()
        return self.concatenated_matrices