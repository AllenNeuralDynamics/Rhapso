import pandas as pd
import numpy as np

"""
View Transform Models parses and combines view registrations matrices
"""

class ViewTransformModels:
    def __init__(self, df):
        self.view_registrations_df = df.get("view_registrations", pd.DataFrame())
        self.calibration_matrices = {}
        self.rotation_matrices = {}
        self.concatenated_matrices = {}

    def create_transform_matrices(self):
        """
        Extracts transformation matrices from a dataframe and organizes them into appropriate data structures
        based on their types and intended usage.
        """
        if self.view_registrations_df.empty: raise ValueError("view_registrations_df is empty")
        
        # parse DF for view_transform matrices
        for _, row in self.view_registrations_df.iterrows():
            if row["type"] == "affine":

                # create affine matrix
                affine_values = np.fromstring(row["affine"], sep=",").astype(np.float64)
                if len(affine_values) == 12:
                    affine_values = np.append(affine_values, [0, 0, 0, 1])  # append homogeneous coordinates
                affine_matrix = affine_values.reshape(4, 4)

                # append matrix by row name
                view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"
                if "calibration" in row["name"].lower():
                    self.calibration_matrices[view_id] = {"affine_matrix": affine_matrix}
                else:
                    self.rotation_matrices[view_id] = {"affine_matrix": affine_matrix}

    def concatenate_matrices_by_view_id(self):
        """
        Concatenates calibration and rotation matrices for each view ID, if available.
        """
        if not self.calibration_matrices and not self.rotation_matrices: raise ValueError("No matrices to concatenate")
        
        # Zarr
        if not self.calibration_matrices:
            self.concatenated_matrices = {
                key: self.rotation_matrices[key]["affine_matrix"]
                for key in self.rotation_matrices
            }
        
        # TIFF
        else:
            for key in self.calibration_matrices:
                if key in self.rotation_matrices:
                    calibration_matrix = self.calibration_matrices[key]["affine_matrix"]
                    rotation_matrix = self.rotation_matrices[key]["affine_matrix"]
                    concatenated_matrix = np.dot(rotation_matrix, calibration_matrix)
                    self.concatenated_matrices[key] = concatenated_matrix

    def run(self):
        """
        Executes the entry point of the script.
        """
        self.create_transform_matrices()
        self.concatenate_matrices_by_view_id()
        return self.concatenated_matrices
