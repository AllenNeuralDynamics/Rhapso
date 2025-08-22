import pandas as pd
import numpy as np

"""
Utility class to parse and combine view registrations matrices
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
        if self.view_registrations_df.empty:
            raise ValueError("view_registrations_df is empty")

        for _, row in self.view_registrations_df.iterrows():
            if row["type"] == "affine":
                # Create affine matrix
                affine_values = np.fromstring(row["affine"], sep=",").astype(np.float64)
                if len(affine_values) == 12:
                    affine_values = np.append(affine_values, [0, 0, 0, 1])  # append homogeneous row
                affine_matrix = affine_values.reshape(4, 4)

                # Get view ID
                view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"
                name = row["name"].strip().lower()

                # Store matrix based on type
                if "translation to nominal grid" in name:
                    self.calibration_matrices.setdefault(view_id, []).append(affine_matrix)
                else:
                    self.rotation_matrices.setdefault(view_id, []).append(affine_matrix)

    def concatenate_matrices_by_view_id(self):
        """
        Concatenates calibration and rotation matrices for each view ID, if available.
        """
        if not self.calibration_matrices and not self.rotation_matrices:
            raise ValueError("No matrices to concatenate")

        all_keys = set(self.calibration_matrices.keys()).union(self.rotation_matrices.keys())

        for key in all_keys:
            rotation_matrices = self.rotation_matrices.get(key, [])
            translation_matrices = self.calibration_matrices.get(key, [])

            # First combine all rotation matrices
            rotation_combined = np.eye(4)
            for mat in rotation_matrices:
                rotation_combined = np.dot(rotation_combined, mat)

            # Then combine all translation matrices
            translation_combined = np.eye(4)
            for mat in translation_matrices:
                translation_combined = np.dot(translation_combined, mat)

            # Final = translation * rotation (apply rotation first, then translation)
            final_matrix = np.dot(rotation_combined, translation_combined)

            self.concatenated_matrices[key] = final_matrix

    def run(self):
        """
        Executes the entry point of the script.
        """
        self.create_transform_matrices()
        self.concatenate_matrices_by_view_id()
        return self.concatenated_matrices
