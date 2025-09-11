import numpy as np
import copy
import math

"""
GlobalOptimization iteratively refines per-tile transforms to achieve sub-pixel alignment using matched point correspondences. 
"""

class GlobalOptimization:
    def __init__(self, tiles, relative_threshold, absolute_threshold, min_matches, 
                 damp, max_iterations, max_allowed_error, max_plateauwidth, run_type, metrics_output_path):
        self.tiles = tiles
        self.relative_threshold = relative_threshold
        self.absolute_threshold = absolute_threshold
        self.min_matches = min_matches
        self.damp = damp
        self.max_iterations = max_iterations
        self.max_allowed_error = max_allowed_error
        self.max_plateauwidth = max_plateauwidth
        self.run_type = run_type
        self.metrics_output_path = metrics_output_path
        self.validation_stats = {
            'solve_metrics_per_tile': {
                'i': 0,
                'stats': []
            }
        }
        self.observer = {
            'max': 0,
            'mean': 0,
            'median': 0,
            'min': float('inf'),
            'slope': [],
            'sorted_values': [],
            'square_differences': 0,
            'squares': 0,
            'std': 0,
            'std_0': 0,
            'values': [],
            'var': 0,
            'var_0': 0,     
        }
        # self.save_metrics = JSONFileHandler(self.metrics_output_path)
    
    def update_observer(self, new_value):
        obs = self.observer

        obs['values'].append(new_value)
        obs['sorted_values'].sort()

        if len(obs['values']) == 1:
            obs['slope'].append(0.0)
            obs['mean'] = new_value
            obs['var'] = 0
            obs['var_0'] = 0
        else:
            obs['slope'].append(new_value - obs['values'][-2])
            
            delta = new_value - obs['mean']
            obs['mean'] += delta / (len(obs['values']))

            obs['square_differences'] += delta * (new_value - obs['mean'])
            obs['var'] = obs['square_differences'] / (len(obs['values']) - 1)

            obs['squares'] += new_value * new_value
            obs['var_0'] = obs['squares'] / (len(obs['values']) - 1)

        obs['std_0'] = math.sqrt(obs['var_0'])
        obs['std'] = math.sqrt(obs['var'])

        if new_value < obs['min']:
            obs['min'] = new_value
        if new_value > obs['max']:
            obs['max'] = new_value

    def update_cost(self, tile):
        """
        Computes and stores the average distance and weighted cost (fit quality) of point matches for a tile.
        """
        distance = 0.0
        cost = 0.0
        if len(tile["matches"]) > 0:
            sum_weight = 0.0

            for match in tile["matches"]:
                dl = np.linalg.norm(np.array(match["p1"]["w"]) - np.array(match["p2"]["w"]))

                distance += dl
                cost += dl * dl * match['weight']
                sum_weight += match['weight']

            distance /= len(tile["matches"])
            cost /= sum_weight
        
        tile['model']['cost'] = cost
        tile['cost'] = cost
        tile['distance'] = distance

    def update_errors(self):
        """
        Monitor convergence by updating cost metrics for all tiles and returns the average alignment error.
        """
        total_distance = 0.0
        min_error = float("inf")
        max_error = 0.0

        for tile in self.tiles:
            self.update_cost(tile)

            if tile['distance'] < min_error:
                min_error = tile['distance']
            if tile['distance'] > max_error:
                max_error = tile['distance']
            total_distance += tile['distance']

        average_error = total_distance / len(self.tiles)  

        # self.save_metrics.update(
        #     "alignment errors",
        #     {
        #         "min_error": min_error,
        #         "max_error": max_error,
        #         "mean_error": average_error,
        #     },
        # )  
        
        return average_error 

    def apply_damp(self, tile):
        """
        Use model to align p1 in all tile point matches
        """
        model = tile["model"]["regularized"]
        matches = tile["matches"]

        for match in matches:
            a = self.apply_model_in_place(copy.deepcopy(match['p1']['l']), model)

            for i in range(len(a)):
                match['p1']['w'][i] += self.damp * (a[i] - match['p1']['w'][i])

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
        Affine transformation model updating using scalar math.
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

    def fit(self, tile):
        """
        Fits multiple transformation models to a tile.
        """
        affine = self.affine_fit_model(tile['model']['a'], tile['matches'])
        rigid = self.rigid_fit_model(tile['model']['b'], tile['matches'])
        regularized = self.regularize_models(affine, rigid)
        
        tile['model']['a'] = affine
        tile['model']['b'] = rigid
        tile['model']['regularized'] = regularized
    
    def apply_model_in_place(self, point, model):
        x, y, z = point[0], point[1], point[2]
        point[0] = model['m00'] * x + model['m01'] * y + model['m02'] * z + model['m03']
        point[1] = model['m10'] * x + model['m11'] * y + model['m12'] * z + model['m13']
        point[2] = model['m20'] * x + model['m21'] * y + model['m22'] * z + model['m23']

        return point
    
    def apply(self):     
        for tile in self.tiles:
            if self.run_type == 'affine' or self.run_type == 'split-affine':
                model = tile['model']['regularized']
            elif self.run_type == 'rigid':
                model = tile['model']['b']
            
            for match in tile['matches']:
                match['p1']['w'][:] = match['p1']['l']
                match['p1']['w'][:] = self.apply_model_in_place(match['p1']['w'], model)
    
    def get_wide_slope(self, values, width):
        width = int(width)
        return (values[-1] - values[-1 - width]) / width

    def optimize_silently(self):
        """
        Iteratively refines tile alignments using model fitting and dampening until convergence or max iterations.
        """
        i = 0
        proceed = i < self.max_iterations
        self.apply()

        while proceed:
            if not self.tiles:
                return
            
            for tile in self.tiles:         
                self.fit(tile)
                self.apply_damp(tile)

            error = self.update_errors()
            self.update_observer(error)
            self.validation_stats.setdefault('solver_metrics_per_tile', {}).setdefault('stats', []).append({
                'iteration': i,
                'observer': copy.deepcopy(self.observer),
            })

            if i > self.max_plateauwidth:
                proceed = error > self.max_allowed_error
                d = self.max_plateauwidth

                while not proceed and d >= 1:
                    proceed = proceed or abs(self.get_wide_slope(self.observer['values'], d)) > 0.0001
                    d /= 2
            
            i += 1
            if i >= self.max_iterations:
                proceed = False

    def run(self):
        """
        Executes the entry point of the script.
        """
        self.optimize_silently()

        return self.tiles, self.validation_stats