import numpy as np

"""
Compute Tiles prepares a set of tiles and pairwise correspondences for global alignment.
"""

class ComputeTiles:
    def __init__(self, pmc, view_id_set, groups, dataframes, run_type):
        self.pmc = pmc
        self.view_id_set = view_id_set
        self.groups = groups
        self.dataframes = dataframes
        self.run_type = run_type
    
    def flip_matches(self, matches):
        """
        Swap endpoints of each match to create the reverse (B→A) correspondences
        """
        flipped = []
        for match in matches:
            p1 = match['p2']
            p2 = match['p1']
            weight = match.get('weight', 1)
            strength = match.get('strength', 1)

            flipped.append({
                'p1': p1,
                'p2': p2,
                'weight': weight,
                'strength': strength
            })
        return flipped
    
    def get_bounding_boxes(self, M, dims):
        """
        Compute world-space AABB (min/max corners) of a voxel-aligned box of size dims after applying affine M
        """
        M = np.asarray(M, float)
        if M.shape == (3, 4):
            M = np.vstack([M, [0.0, 0.0, 0.0, 1.0]])

        # interval mins/maxes
        t0 = 0.0; t1 = 0.0; t2 = 0.0
        s0 = float(dims[0]) - 1.0
        s1 = float(dims[1]) - 1.0
        s2 = float(dims[2]) - 1.0

        A = M[:3, :3]
        tx, ty, tz = M[0, 3], M[1, 3], M[2, 3]

        # row 0
        tt0 = A[0,0]*t0 + A[0,1]*t1 + A[0,2]*t2 + tx
        rMin0 = rMax0 = tt0
        rMin0 += s0*A[0,0] if A[0,0] < 0 else 0.0; rMax0 += 0.0 if A[0,0] < 0 else s0*A[0,0]
        rMin0 += s1*A[0,1] if A[0,1] < 0 else 0.0; rMax0 += 0.0 if A[0,1] < 0 else s1*A[0,1]
        rMin0 += s2*A[0,2] if A[0,2] < 0 else 0.0; rMax0 += 0.0 if A[0,2] < 0 else s2*A[0,2]

        # row 1
        tt1 = A[1,0]*t0 + A[1,1]*t1 + A[1,2]*t2 + ty
        rMin1 = rMax1 = tt1
        rMin1 += s0*A[1,0] if A[1,0] < 0 else 0.0; rMax1 += 0.0 if A[1,0] < 0 else s0*A[1,0]
        rMin1 += s1*A[1,1] if A[1,1] < 0 else 0.0; rMax1 += 0.0 if A[1,1] < 0 else s1*A[1,1]
        rMin1 += s2*A[1,2] if A[1,2] < 0 else 0.0; rMax1 += 0.0 if A[1,2] < 0 else s2*A[1,2]

        # row 2
        tt2 = A[2,0]*t0 + A[2,1]*t1 + A[2,2]*t2 + tz
        rMin2 = rMax2 = tt2
        rMin2 += s0*A[2,0] if A[2,0] < 0 else 0.0; rMax2 += 0.0 if A[2,0] < 0 else s0*A[2,0]
        rMin2 += s1*A[2,1] if A[2,1] < 0 else 0.0; rMax2 += 0.0 if A[2,1] < 0 else s1*A[2,1]
        rMin2 += s2*A[2,2] if A[2,2] < 0 else 0.0; rMax2 += 0.0 if A[2,2] < 0 else s2*A[2,2]

        rMin = np.array([rMin0, rMin1, rMin2], float)
        rMax = np.array([rMax0, rMax1, rMax2], float)
        return rMin, rMax

    def bounding_boxes(self, M, dims):
        """
        Compute an integer, padded axis-aligned bounding box from the real-valued bounds given transform M and volume size dims
        """
        rMin, rMax = self.get_bounding_boxes(M, dims)
        min_i = np.rint(rMin).astype(int) - 1
        max_i = np.rint(rMax).astype(int) + 1
        return (min_i.tolist(), max_i.tolist())
    
    def transform_matrices(self, view): 
        """
        Compose the 4x4 world transform for a view by fetching all its affine models and chaining them in order
        """
        M = np.eye(4, dtype=float) 

        view, setup = [p.strip() for p in view['view'].split(", ", 1)]      
        view_key   = int(view.split(": ", 1)[1])                       
        setup_key = int(setup.split(": ", 1)[1])     
        
        vr_df = self.dataframes["view_registrations"]
        sub = vr_df[
            (vr_df["timepoint"].astype(int) == view_key) &
            (vr_df["setup"].astype(int) == setup_key) &
            (vr_df["type"] == "affine")
        ]

        for model in sub["affine"]:   
            vals = np.fromstring(model.replace(",", " "), sep=" ", dtype=float)
            T = np.eye(4, dtype=float); T[:3, :4] = vals.reshape(3, 4)  
            M = M @ T

        return M
    
    def overlaps(self, bba, bbb):
        """
        Boolean check if two axis-aligned boxes *strictly* overlap on every axis
        """
        (minA, maxA) = bba
        (minB, maxB) = bbb
        for d in range(len(minA)):  
            if ((minA[d] <= minB[d] and maxA[d] <= minB[d]) or
                (minA[d] >= maxB[d] and maxA[d] >= maxB[d])):
                return False
        return True

    def overlap(self, view_a, dims_a, view_b, dims_b):
        """
        Builds each view transform, computes its AABB, then checks overlap
        """
        ma = self.transform_matrices(view_a)
        mb = self.transform_matrices(view_b)

        bba = self.bounding_boxes(ma, dims_a)
        bbb = self.bounding_boxes(mb, dims_b)

        return self.overlaps(bba, bbb) 
    
    def bb_overlap(self, real_bb1, real_bb2):
        """
        Axis-aligned box overlap test
        """
        min1, max1 = real_bb1
        min2, max2 = real_bb2
        for d in range(len(min1)):
            if (min1[d] < min2[d] and max1[d] < min2[d]) or (min1[d] > max2[d] and max1[d] > max2[d]):
                return False
        
        return True
    
    def get_overlap_interval(self, view_a, dims_a, view_b, dims_b):
        """
        Compute the continuous overlap box in world space between views A and B
        """
        ma = self.transform_matrices(view_a)
        mb = self.transform_matrices(view_b)

        bb1 = self.get_bounding_boxes(ma, dims_a)
        bb2 = self.get_bounding_boxes(mb, dims_b)
        
        if not bb1 or not bb2:
            return 

        real_bb1 = self.bounding_boxes(ma, dims_a)
        real_bb2 = self.bounding_boxes(mb, dims_b)

        if self.bb_overlap(real_bb1, real_bb2):
            rmin1, rmax1 = real_bb1
            rmin2, rmax2 = real_bb2

            mins = [0.0] * len(rmin1)
            maxs = [0.0] * len(rmin1)

            for d in range(len(rmin1)):
                mins[d] = max(rmin1[d], rmin2[d])
                maxs[d] = min(rmax1[d], rmax2[d])

                if d == 2 and mins[d] == maxs[d] == 0.0 and dims_a[2] == 1 and dims_b[2] == 1:
                    mins[d], maxs[d] = 0.0, 1.0
                elif mins[d] == maxs[d] or maxs[d] < mins[d]:
                    return None
            
            return (mins, maxs)
    
    def cube_for(self, overlap):
        """
        Find the 8 corner points of an axis-aligned 3D box
        """
        mins, maxs = overlap
        min0, min1, min2 = map(float, mins)
        max0, max1, max2 = map(float, maxs)

        return [
            [min0, min1, min2],
            [min0, min1, max2],
            [min0, max1, min2],
            [min0, max1, max2],
            [max0, min1, min2],
            [max0, min1, max2],
            [max0, max1, min2],
            [max0, max1, max2],
        ]
    
    def apply(self, model, source, target):
        """
        Apply a 3D affine transform
        """
        x, y, z = float(source[0]), float(source[1]), float(source[2])

        if isinstance(model, dict):
            t0 = x*model['m00'] + y*model['m01'] + z*model['m02'] + model['m03']
            t1 = x*model['m10'] + y*model['m11'] + z*model['m12'] + model['m13']
            t2 = x*model['m20'] + y*model['m21'] + z*model['m22'] + model['m23']
        else:
            M = np.asarray(model, float)
            if M.shape == (3, 4) or M.shape == (4, 4):
                t0 = x*M[0,0] + y*M[0,1] + z*M[0,2] + M[0,3]
                t1 = x*M[1,0] + y*M[1,1] + z*M[1,2] + M[1,3]
                t2 = x*M[2,0] + y*M[2,1] + z*M[2,2] + M[2,3]
            else:
                raise ValueError("model must be dict m00..m23 or a 3x4/4x4 array")

        target[0] = t0
        target[1] = t1
        target[2] = t2

        return target
    
    def assign_weak_link_point_matches(self, view_map, groups):
        """
        Create "weak-link" synthetic matches between tiles that belong to different groups but spatially overlap, then attach 
        those matches to all tiles in the two groups and connect the tile graphs (without duplicate edges)
        """
        group_map = {}
        for v in view_map: 
            for group in groups:
                if v in group['views']:
                    group_map[v] = group
                    break
        
        views = list(view_map.values())
        views = sorted(views, key=lambda d: int(d['view'].partition('setup:')[2].strip().split()[0]))
        for a in range(len(views) - 1):
            for b in range(a + 1, len(views)):
                view_a = views[a]
                view_b = views[b]

                if group_map[view_a['view']] == group_map[view_b['view']]:
                    continue

                pm = []

                vs_df = self.dataframes["view_setups"]
                vs_df["id"] = vs_df["id"].astype(int)
                vs_df = vs_df[vs_df["name"].isna()]
                vs_idx = vs_df.assign(id_int=vs_df["id"].astype(int)).set_index("id_int")

                setup_a = int(view_a['view'].split(", setup: ", 1)[1])
                setup_b = int(view_b["view"].split(", setup: ", 1)[1])

                row_a = vs_idx.loc[setup_a]
                row_b = vs_idx.loc[setup_b]

                dims_a = tuple(map(int, str(row_a["size"]).split()))
                dims_b = tuple(map(int, str(row_b["size"]).split()))
            
                if self.overlap(view_a, dims_a, view_b, dims_b):
                    
                    overlap = self.get_overlap_interval(view_a, dims_a, view_b, dims_b)

                    if overlap is None:
                        continue

                    pa = self.cube_for(overlap)
                    pb = self.cube_for(overlap)

                    key = (lambda v: v if isinstance(v, str) else v["view"])
                    ta = next(m["model"]["regularized"] for m in self.pmc["models"] if m["view"] == key(view_a))
                    tb = next(m["model"]["regularized"] for m in self.pmc["models"] if m["view"] == key(view_b))

                    for i in range(len(pa)):
                        points_a = self.apply(ta, pa[i], pa[i])
                        points_b = self.apply(tb, pb[i], pb[i])
                        if points_a is None or points_b is None:
                            print()
                        match = {
                            "p1": {
                                "l": points_a,   
                                "w": points_a,    
                                "weight": 1,
                                "strength": 1
                            },
                            "p2": {
                                "l": points_b,
                                "w": points_b,
                                "weight": 1,
                                "strength": 1
                            },
                            "weight": 1,
                            "strength": 1
                        }
                        pm.append(match)
                    

                    idx = next((i for i, g in enumerate(groups) if view_a['view'] in g.get('views', ())), None)
                    views_a = groups[idx]['views'] if idx is not None else [view_a['view']]

                    for va in views_a:
                        tile_a = view_map.get(va)
                        if tile_a:
                            tile_a['matches'].extend(pm)

                    flipped_matches = self.flip_matches(pm)
                    idx = next((i for i, g in enumerate(groups) if view_b['view'] in g.get('views', ())), None)
                    views_b = groups[idx]['views'] if idx is not None else [view_b['view']]
                    
                    for vb in views_b:
                        tile_b = view_map.get(vb)
                        if tile_b:
                            tile_b['matches'].extend(flipped_matches)

                    # Precompute tile lists for both groups
                    tiles_a = [view_map[va] for va in views_a if va in view_map]
                    tiles_b = [view_map[vb] for vb in views_b if vb in view_map]

                    # Initialize a fast membership set on each tile once
                    from itertools import chain
                    for t in chain(tiles_a, tiles_b):
                        ct = t.setdefault('connected_tiles', [])
                        s = t.get('_connected_set')
                        if s is None:
                            # normalize existing entries (dict or str) to view-id strings
                            s = {(c['view'] if isinstance(c, dict) else c) for c in ct}
                            t['_connected_set'] = s

                    views_a_set = {t['view'] for t in tiles_a}
                    views_b_set = {t['view'] for t in tiles_b}

                    # A -> B (add only what’s missing)
                    for ta in tiles_a:
                        missing = views_b_set - ta['_connected_set'] - {ta['view']}
                        if missing:
                            ta['connected_tiles'].extend({'view': vb, 'tile': view_map.get(vb)} for vb in missing)
                            ta['_connected_set'].update(missing)

                    # B -> A (add only what’s missing)
                    for tb in tiles_b:
                        missing = views_a_set - tb['_connected_set'] - {tb['view']}
                        if missing:
                            tb['connected_tiles'].extend({'view': va, 'tile': view_map.get(va)} for va in missing)
                            tb['_connected_set'].update(missing)
                            
        return view_map
    
    def assign_point_matches(self, map):
        """
        Attach inlier correspondences to each tile for both directions
        """
        for pair in self.pmc:
            pair_a = pair['view'][0]
            pair_b = pair['view'][1]
            tile_a = map[pair_a]
            tile_b = map[pair_b]

            correspondences = pair['inliers']
            if len(correspondences) > 0:

                pm = correspondences
                flipped_matches = self.flip_matches(pm)

                tile_a['matches'].extend(pm)
                tile_b['matches'].extend(flipped_matches)

                tile_a['connected_tiles'].append({'view': pair_b, 'tile': tile_b})
                tile_b['connected_tiles'].append({'view': pair_a, 'tile': tile_a})
                
                pair['flipped'] = flipped_matches
        
        return map

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
        return {
            'a' : self.create_default_model_3d(),
            'b' : self.create_default_model_3d(),
            'regularized': self.create_default_model_3d(),
            'cost' : 1.7976931348623157e+308,
            'l1' : 0.900000,
            'lambda' : 0.100000
        }

    def assign_views_to_tiles(self, groups):
        """
        Create initial view_map entry for each view to be optimized.
        """
        view_map = {}   
        if groups:
            
            remaining_views = {f"timepoint: {tp}, setup: {vs}" for (tp, vs) in self.view_id_set}
            for group in groups:
                for view in group['views']:
                    view_map[view] = {
                        'view': view,
                        'connected_tiles': [],
                        'cost': 0,
                        'distance': 0,
                        'matches': [],
                        'model': self.create_models()
                    }
            
                    if view not in remaining_views:
                        raise RuntimeError(f"{view} is part of two groups; groups should have been merged.")
                
                    remaining_views.remove(view)
            
            for view in remaining_views:
                view_map[view] = {
                    'views': [view],
                    'connected_tiles': [],
                    'cost': 0,
                    'distance': 0,
                    'matches': [],
                    'model': self.create_models(),
                }
        
        else:
            for view in self.view_id_set:
                tp, setup = view
                key = f"timepoint: {tp}, setup: {setup}"

                view_map[key] = {
                    'view': key,
                    'connected_tiles': [],
                    'cost': 0,
                    'distance': 0,
                    'matches': [],
                    'model': self.create_models()
                }
            
        return view_map
    
    def merge_all_overlapping_groups(self):
        """
        Repeatedly merge any groups that share at least one view until no overlaps remain
        """
        g = [{'views': list(gr.get('views', []))} for gr in self.groups]

        while True:
            pair = None
            n = len(g)

            for a in range(n - 1):
                va = set(g[a]['views'])
                for b in range(a + 1, n):
                    if va & set(g[b]['views']):  # overlaps?
                        pair = (a, b)
                        break
                if pair:
                    break

            if not pair:
                break

            i, j = pair
            ga, gb = g[i], g[j]

            # remove indexB then indexA (j > i)
            del g[j]
            del g[i]

            # merge(ga, gb): preserve order, dedup
            merged_views = list(dict.fromkeys(ga['views'] + gb['views']))
            g.append({'views': merged_views})

        return g

    def init_global_opt(self):
        """
        Build the tile map and attach point matches 
        """
        if self.groups is None:
            groups = self.groups
        else:
            groups = self.groups

        view_map = self.assign_views_to_tiles(groups) 

        if self.groups is None:
            view_map = self.assign_point_matches(view_map)
        else:
            view_map = self.assign_weak_link_point_matches(view_map, groups)

        return view_map
    
    def add_and_fix_tiles(self, view_map):
        """
        Build the initial tile collection for alignment
        """
        tc = {
            'error': 0,
            'fixed_tiles': [],
            'max_error': 0,
            'min_error': float('inf'),
            'tiles': []
        }

        if self.groups:
            view_to_group_idx = {v: gi for gi, g in enumerate(self.groups) for v in g.get('views', [])}

            first_by_group = {}
            for view_id, gi in view_to_group_idx.items():
                first_by_group.setdefault(gi, view_id)

            # add exactly one tile per group 
            for gi in sorted(first_by_group):
                rep_view = first_by_group[gi]
                t = view_map.get(rep_view)
                if len(t['connected_tiles']) > 0:
                    tc['tiles'].append(t)

        else:
            tiles = []
            for tp, setup in self.view_id_set:
                key = f"timepoint: {tp}, setup: {setup}"
                tile = view_map[key]      
                tiles.append(tile)
            
            for tile in tiles:
                if len(tile['connected_tiles']) > 0:
                    tc['tiles'].append(tile)
            
        return tc
    
    def compute_tiles(self):
        """
        Interface tile computing
        """
        view_map = self.init_global_opt()
        tc = self.add_and_fix_tiles(view_map)

        if len(tc['tiles']) == 0:
            return None
        else:
            return tc, view_map

    def run(self):
        """
        Executes the entry point of the script.
        """
        tc, view_map = self.compute_tiles()
        return tc, view_map