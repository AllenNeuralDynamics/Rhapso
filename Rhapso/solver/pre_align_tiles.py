import numpy as np
import random

"""
Pre Align Tiles roughly align p1 with p2 to speed up global optimization rounds
"""

class PreAlignTiles:
    def __init__(self, min_matches, run_type, fixed_tile):
        self.min_matches = min_matches
        self.run_type = run_type
        self.fixed_tile = fixed_tile
        
    def rigid_fit_model(self, rigid_model, matches):
        """
        Computes the best-fit rigid transformation (rotation + translation)
        using unweighted quaternion-based estimation between 3D point sets.
        """

        # === Compute unweighted centroids ===
        pc = np.mean([m['p1']['l'] for m in matches], axis=0)
        qc = np.mean([m['p2']['w'] for m in matches], axis=0)

        # === Accumulate scalar components of S matrix ===
        Sxx = Sxy = Sxz = Syx = Syy = Syz = Szx = Szy = Szz = 0.0

        for m in matches:
            px, py, pz = m['p1']['l'] - pc
            qx, qy, qz = m['p2']['w'] - qc

            Sxx += px * qx
            Sxy += px * qy
            Sxz += px * qz
            Syx += py * qx
            Syy += py * qy
            Syz += py * qz
            Szx += pz * qx
            Szy += pz * qy
            Szz += pz * qz

        # === Construct symmetric matrix N ===
        N = np.array([
            [Sxx + Syy + Szz, Syz - Szy,         Szx - Sxz,         Sxy - Syx],
            [Syz - Szy,       Sxx - Syy - Szz,   Sxy + Syx,         Szx + Sxz],
            [Szx - Sxz,       Sxy + Syx,        -Sxx + Syy - Szz,   Syz + Szy],
            [Sxy - Syx,       Szx + Sxz,         Syz + Szy,        -Sxx - Syy + Szz]
        ])

        if not np.all(np.isfinite(N)):
            raise ValueError("Matrix N contains NaNs or Infs")

        # === Eigenvalue decomposition ===
        eigenvalues, eigenvectors = np.linalg.eigh(N)
        q = eigenvectors[:, np.argmax(eigenvalues)]
        q /= np.linalg.norm(q)
        q0, qx, qy, qz = q

        # === Quaternion to rotation matrix ===
        R = np.array([
            [q0*q0 + qx*qx - qy*qy - qz*qz,     2*(qx*qy - q0*qz),         2*(qx*qz + q0*qy)],
            [2*(qy*qx + q0*qz),         q0*q0 - qx*qx + qy*qy - qz*qz,     2*(qy*qz - q0*qx)],
            [2*(qz*qx - q0*qy),         2*(qz*qy + q0*qx),         q0*q0 - qx*qx - qy*qy + qz*qz]
        ])

        # === Translation ===
        t = qc - R @ pc

        # === Populate model ===
        rigid_model['m00'], rigid_model['m01'], rigid_model['m02'] = R[0, :]
        rigid_model['m10'], rigid_model['m11'], rigid_model['m12'] = R[1, :]
        rigid_model['m20'], rigid_model['m21'], rigid_model['m22'] = R[2, :]
        rigid_model['m03'], rigid_model['m13'], rigid_model['m23'] = t

        return rigid_model
    
    def affine_fit_model(self, affine_model, matches):
        """
        Exact translation of the Java affine fit() method into Python using scalar math.
        """

        if len(matches) < 3:
            raise ValueError("Not enough matches for affine fit")

        # === Centroids ===
        pcx = pcy = pcz = 0.0
        qcx = qcy = qcz = 0.0
        for m in matches:
            p = m['p1']['l']
            q = m['p2']['w']
            pcx += p[0]
            pcy += p[1]
            pcz += p[2]
            qcx += q[0]
            qcy += q[1]
            qcz += q[2]
        n = len(matches)
        pcx /= n
        pcy /= n
        pcz /= n
        qcx /= n
        qcy /= n
        qcz /= n

        # === Accumulate A and B ===
        a00 = a01 = a02 = a11 = a12 = a22 = 0.0
        b00 = b01 = b02 = b10 = b11 = b12 = b20 = b21 = b22 = 0.0

        for m in matches:
            p = m['p1']['l']
            q = m['p2']['w']
            px = p[0] - pcx
            py = p[1] - pcy
            pz = p[2] - pcz
            qx = q[0] - qcx
            qy = q[1] - qcy
            qz = q[2] - qcz

            a00 += px * px
            a01 += px * py
            a02 += px * pz
            a11 += py * py
            a12 += py * pz
            a22 += pz * pz

            b00 += px * qx
            b01 += px * qy
            b02 += px * qz
            b10 += py * qx
            b11 += py * qy
            b12 += py * qz
            b20 += pz * qx
            b21 += pz * qy
            b22 += pz * qz

        # === Compute inverse of A manually ===
        det = (
            a00 * a11 * a22 +
            a01 * a12 * a02 +
            a02 * a01 * a12 -
            a02 * a11 * a02 -
            a12 * a12 * a00 -
            a22 * a01 * a01
        )

        if det == 0:
            raise ValueError("Affine matrix is singular")

        idet = 1.0 / det
        ai00 = (a11 * a22 - a12 * a12) * idet
        ai01 = (a02 * a12 - a01 * a22) * idet
        ai02 = (a01 * a12 - a02 * a11) * idet
        ai11 = (a00 * a22 - a02 * a02) * idet
        ai12 = (a02 * a01 - a00 * a12) * idet
        ai22 = (a00 * a11 - a01 * a01) * idet

        # === Compute transformation matrix ===
        m00 = ai00 * b00 + ai01 * b10 + ai02 * b20
        m01 = ai01 * b00 + ai11 * b10 + ai12 * b20
        m02 = ai02 * b00 + ai12 * b10 + ai22 * b20

        m10 = ai00 * b01 + ai01 * b11 + ai02 * b21
        m11 = ai01 * b01 + ai11 * b11 + ai12 * b21
        m12 = ai02 * b01 + ai12 * b11 + ai22 * b21

        m20 = ai00 * b02 + ai01 * b12 + ai02 * b22
        m21 = ai01 * b02 + ai11 * b12 + ai12 * b22
        m22 = ai02 * b02 + ai12 * b12 + ai22 * b22

        m03 = qcx - m00 * pcx - m01 * pcy - m02 * pcz
        m13 = qcy - m10 * pcx - m11 * pcy - m12 * pcz
        m23 = qcz - m20 * pcx - m21 * pcy - m22 * pcz

        # === Assign ===
        affine_model['m00'], affine_model['m01'], affine_model['m02'], affine_model['m03'] = m00, m01, m02, m03
        affine_model['m10'], affine_model['m11'], affine_model['m12'], affine_model['m13'] = m10, m11, m12, m13
        affine_model['m20'], affine_model['m21'], affine_model['m22'], affine_model['m23'] = m20, m21, m22, m23

        return affine_model
    
    def regularize_models(self, affine, rigid):
        """
        Blend affine and rigid models into a single "regularized" 3x4 affine by convex combination 
        (90% affine, 10% rigid)
        """
        alpha=0.1
        l1 = 1.0 - alpha

        def to_array(model):
            return [
                model['m00'], model['m01'], model['m02'], model['m03'], 
                model['m10'], model['m11'], model['m12'], model['m13'],  
                model['m20'], model['m21'], model['m22'], model['m23'], 
            ]

        afs = to_array(affine)
        bfs = to_array(rigid)
        rfs = [l1 * a + alpha * b for a, b in zip(afs, bfs)]

        keys = [
            'm00', 'm01', 'm02', 'm03',
            'm10', 'm11', 'm12', 'm13',
            'm20', 'm21', 'm22', 'm23',
        ]
        regularized = dict(zip(keys, rfs))

        return regularized

    def fit(self, tile, pm):
        """
        Fits multiple transformation models to a tile using provided point matches.
        """
        affine = self.affine_fit_model(tile['model']['a'], pm)
        rigid = self.rigid_fit_model(tile['model']['b'], pm)
        regularized = self.regularize_models(affine, rigid)
        
        tile['model']['a'] = affine
        tile['model']['b'] = rigid
        tile['model']['regularized'] = regularized

        return tile
    
    def get_connected_point_matches(self, target_tile, reference_tile):
        """
        Finds point matches in the target tile that connect to the reference tile.
        """
        reference_point_ids = {id(match['p1']) for match in reference_tile['matches']}

        # Collect matches in the target tile that connect to any reference point by object identity
        connected_point_matches = [
            match for match in target_tile['matches']
            if id(match['p2']) in reference_point_ids
        ]

        return connected_point_matches

    def apply_model_in_place(self, point, model):
        x, y, z = point[0], point[1], point[2]
        point[0] = model['m00'] * x + model['m01'] * y + model['m02'] * z + model['m03']
        point[1] = model['m10'] * x + model['m11'] * y + model['m12'] * z + model['m13']
        point[2] = model['m20'] * x + model['m21'] * y + model['m22'] * z + model['m23']

        return point
    
    def apply_transform_to_tile(self, tile):  
        if self.run_type == "affine" or self.run_type == "split-affine":
            model = tile['model']['regularized'] 
        elif self.run_type == "rigid":
            model = tile['model']['b'] 
        
        for match in tile['matches']:
            match['p1']['w'][:] = match['p1']['l']
            self.apply_model_in_place(match['p1']['w'], model)  
                
    def pre_align(self, tiles):
        """
        Greedily seed an initial alignment
        """
        random.shuffle(tiles['tiles'])

        if getattr(self, "fixed_tile", None):
            seed = next((t for t in tiles['tiles'] if t.get('view') == self.fixed_tile), None)
            if seed is None:
                raise ValueError(f"Fixed tile '{self.fixed_tile}' not found in tiles.")
            tiles['fixed_tiles'] = [seed]

        unaligned_tiles = []
        aligned_tiles = []

        if not tiles:
            return unaligned_tiles, aligned_tiles
        
        if len(tiles['fixed_tiles']) == 0:
            aligned_tiles.append(tiles['tiles'][0])
            unaligned_tiles.extend(tiles['tiles'][1:])
        else:
            for tile in tiles['tiles']:
                if tile in tiles['fixed_tiles']:
                    aligned_tiles.append(tile)
                else:
                    unaligned_tiles.append(tile)
        
        ref_index = 0
        while ref_index < len(aligned_tiles):
            
            if len(unaligned_tiles) == 0:
                break
                
            reference_tile = aligned_tiles[ref_index]
            self.apply_transform_to_tile(reference_tile)
            
            tiles_added = 0
            target_index = 0

            while target_index < len(unaligned_tiles):
                target_tile = unaligned_tiles[target_index]
                
                if any(conn['view'] == target_tile['view'] for conn in reference_tile['connected_tiles']): 
                    pm = self.get_connected_point_matches(target_tile, reference_tile)
                    
                    if len(pm) >= self.min_matches:
                        target_tile = self.fit(target_tile, pm)
                        unaligned_tiles.pop(target_index)
                        aligned_tiles.append(target_tile)
                        tiles_added += 1
                        continue
                
                target_index += 1
            
            # Always move to the next reference tile
            ref_index += 1
        
        return unaligned_tiles

    def run(self, tiles):
        """
        Executes the entry point of the script.
        """
        unaligned_tiles = self.pre_align(tiles)

        if len(unaligned_tiles) > 0:
            print(f"aligned all tiles but: {len(unaligned_tiles)}")

        return tiles['tiles']