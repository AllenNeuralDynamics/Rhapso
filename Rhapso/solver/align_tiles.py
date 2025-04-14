import numpy as np
import copy

# This class updates the model matrices using point matches

class AlignTiles:
    def __init__(self, tiles, pmc, fixed_views):
        self.tiles = tiles
        self.pmc = pmc
        self.fixed_views = fixed_views
        self.min_matches = 3

    def invert_transformation(self, model):
        # Extract the rotation and translation components
        R = np.array([
            [model['m00'], model['m01'], model['m02']],
            [model['m10'], model['m11'], model['m12']],
            [model['m20'], model['m21'], model['m22']]
        ])
        T = np.array([model['m03'], model['m13'], model['m23']])

        # Compute the inverse of the rotation matrix
        R_inv = np.linalg.inv(R)
        
        # Calculate the inverse of the translation using the inverted rotation
        T_inv = -R_inv @ T

        # Update the model with the inverted values
        model['i00'], model['i01'], model['i02'] = R_inv[0, :]
        model['i10'], model['i11'], model['i12'] = R_inv[1, :]
        model['i20'], model['i21'], model['i22'] = R_inv[2, :]
        model['i03'], model['i13'], model['i23'] = T_inv

        return model
    
    def translation_fit_model(self, transformation_matrix, matches):
        pc = np.zeros(3)
        qc = np.zeros(3)
        total_weight = 0.0

        # Calculate the weighted center of mass for p and q
        for match in matches:
            weight = match['weight']
            pc += weight * match['p1']
            qc += weight * match['p2']
            total_weight += weight

        if total_weight == 0:
            raise ValueError("Total weight cannot be zero.")

        pc /= total_weight
        qc /= total_weight

        # Compute the translation
        translation = qc - pc

        # Update the translation part of the transformation matrix
        transformation_matrix[9] = translation[0]
        transformation_matrix[10] = translation[1]
        transformation_matrix[11] = translation[2]

        return transformation_matrix
        
    def rigid_fit_model(self, rigid_model, matches):
        # Calculate weighted centers of mass for p1 and p2
        sumW = sum(m['weight'] for m in matches)
        pc = np.sum([m['weight'] * np.array(m['p1']) for m in matches], axis=0) / sumW
        qc = np.sum([m['weight'] * np.array(m['p2']) for m in matches], axis=0) / sumW

        # Constructing the matrix N from the formula
        S = np.zeros((3, 3))
        for m in matches:
            p_shifted = m['p1'] - pc
            q_shifted = m['p2'] - qc
            w = m['weight']
            S += w * np.outer(p_shifted, q_shifted)

        N = np.zeros((4, 4))
        N[0, 0] = np.trace(S)
        N[1:, 1:] = S + S.T - np.eye(3) * np.trace(S)
        N[0, 1:] = N[1:, 0] = np.array([S[1, 2] - S[2, 1], S[2, 0] - S[0, 2], S[0, 1] - S[1, 0]])

        # Eigenvalue decomposition to find the quaternion representing the rotation
        eigenvalues, eigenvectors = np.linalg.eigh(N)
        q = eigenvectors[:, np.argmax(eigenvalues)]

        # Convert quaternion to rotation matrix
        q0, qx, qy, qz = q
        R = np.array([
            [q0*q0 + qx*qx - qy*qy - qz*qz, 2*(qx*qy - q0*qz), 2*(qx*qz + q0*qy)],
            [2*(qy*qx + q0*qz), q0*q0 - qx*qx + qy*qy - qz*qz, 2*(qy*qz - q0*qx)],
            [2*(qz*qx - q0*qy), 2*(qz*qy + q0*qx), q0*q0 - qx*qx - qy*qy + qz*qz]
        ])

        # Calculate the translation part
        translation = qc - R @ pc

        # Update the rigid model
        rigid_model['m00'], rigid_model['m01'], rigid_model['m02'] = R[0, :]
        rigid_model['m10'], rigid_model['m11'], rigid_model['m12'] = R[1, :]
        rigid_model['m20'], rigid_model['m21'], rigid_model['m22'] = R[2, :]
        rigid_model['m03'], rigid_model['m13'], rigid_model['m23'] = translation

        rigid_model = self.invert_transformation(rigid_model)

        return rigid_model
    
    def affine_fit_model(self, affine_model, matches):
        # Calculate centers of mass for p and q
        pc = np.mean([m['p1'] for m in matches], axis=0)
        qc = np.mean([m['p2'] for m in matches], axis=0)

        # Initialize matrices A and B
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))

        for match in matches:
            p = match['p1']
            q = match['p2']

            p_shifted = p - pc
            q_shifted = q - qc

            # Update matrix A using the outer product of the shifted points
            A += np.outer(p_shifted, p_shifted)

            # Update matrix B using the outer product of p_shifted and q_shifted
            B += np.outer(p_shifted, q_shifted)
        
        # error handling in case identical point matches are found
        if np.linalg.matrix_rank(A) < 3:
            print("matches are too identical")
            return affine_model

        # Solve the normal equations A * X = B for X using matrix inversion
        try:
            Ai = np.linalg.inv(A)
            transformation_matrix = Ai @ B
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix encountered, fitting failed.")

        # Update the affine model matrix elements
        affine_model['m00'], affine_model['m01'], affine_model['m02'] = transformation_matrix[0, :]
        affine_model['m10'], affine_model['m11'], affine_model['m12'] = transformation_matrix[1, :]
        affine_model['m20'], affine_model['m21'], affine_model['m22'] = transformation_matrix[2, :]

        # Recompute translations
        affine_model['m03'] = qc[0] - np.dot(transformation_matrix[0, :], pc)
        affine_model['m13'] = qc[1] - np.dot(transformation_matrix[1, :], pc)
        affine_model['m23'] = qc[2] - np.dot(transformation_matrix[2, :], pc)

        affine_model = self.invert_transformation(affine_model)

        return affine_model

    # Fit models to point matches
    def fit(self, tile, pm):
        tile = copy.deepcopy(tile)
        model = {
            'a': self.affine_fit_model(tile[1]['model']['a'], pm),
            'affine': tile[1]['model']['a'],
            'afs': self.translation_fit_model(tile[1]['model']['afs'], pm),
            'b': self.rigid_fit_model(tile[1]['model']['b'], pm),
            'bfs': self.translation_fit_model(tile[1]['model']['bfs'], pm)
        }
        tile[1]['model'] = model

        return tile
    
    # find point matches between target and reference tile
    def get_connected_point_matches(self, target_tile, reference_tile):
        reference_matches = reference_tile[1]['matches']
        reference_points = []
        
        for match in reference_matches:
            reference_points.append(match['p1'])
        
        connected_point_matches = []
        flipped_point_matches = []

        for match in target_tile[1]['matches']:
            if any(np.array_equal(match['p2'], ref_point) for ref_point in reference_points):
                connected_point_matches.append(match)
                # flipped_match = {'p1': match['p2'], 'p2': match['p1'], 'strength': 1.0, 'weight': 1.0}
                # flipped_point_matches.append(flipped_match)
        
        # implement the flipping process here for the connected point matches
        
        return connected_point_matches
        
    # Interface a pre-alignment process for the transformation matrices
    def pre_align(self):
        aligned_tiles = []
        unaligned_tiles = []
        final_tiles = {}

        # Fixed tiles are considered aligned
        for view, tile in self.tiles.items():
            if view in self.fixed_views:
                aligned_tiles.append((view, tile))
            else:
                unaligned_tiles.append((view, tile))
        
        # Compare all tiles with all tiles and fit transformation models with point matches
        i = 0
        while i < len(aligned_tiles):
            reference_tile = aligned_tiles[i]
            view, tile = reference_tile[0], reference_tile[1]
            
            j = 0
            while j < len(unaligned_tiles):
                target_tile = unaligned_tiles[j]
                parts = target_tile[0].split(',')
                timepoint, setup = parts[0].split()[-1], parts[1].split()[-1]
                label = 'beads'
                view_key = f"{timepoint},{setup},{label}"
                if view_key in reference_tile[1]['connected_tiles']:
                    pm = self.get_connected_point_matches(target_tile, reference_tile)

                    if len(pm) > self.min_matches:
                        aligned_tile = self.fit(target_tile, pm)    
                        aligned_tiles.append(aligned_tile)
                        final_tiles[aligned_tile[0]] = aligned_tile[1]
                        unaligned_tiles.pop(j) 
                        continue  

                j += 1
            i += 1

        return final_tiles, unaligned_tiles  

    def run(self):
        final_tiles, unaligned_tiles = self.pre_align()
        if len(unaligned_tiles) > 0:
            for tile in unaligned_tiles:
                print("the following tiles were not aligned: ")
                print(tile[0])
        return final_tiles