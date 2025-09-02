"""
This class updates tiles for grouping, flipping point matches, and weights
"""

class ComputeTiles:
    def __init__(self, pmc, view_id_set):
        self.pmc = pmc
        self.view_id_set = view_id_set
    
    def flip_matches(self, matches):
        flipped = []
        for match in matches:
            flipped.append({
                'p1': match['p2'],
                'p2': match['p1'],
                'weight': match.get('weight', 1),
                'strength': match.get('strength', 1)
            })
        return flipped
    
    # Add flipped matches to pmc
    def assign_point_matches(self, map):
        for pair in self.pmc:
            pair_a = pair['view'][0]
            pair_b = pair['view'][1]
            tile_a = map[pair_a]
            tile_b = map[pair_b]

            correspondences = pair['inliers']
            if len(correspondences) > 0:

                r = {'compensation': 0, 'sum': 0}

                for pmg in correspondences:
                    # TODO - update weights and r for grouping/split affine
                    pass

                pm = correspondences
                flipped_matches = self.flip_matches(pm)

                tile_a['matches'].extend(pm)
                tile_b['matches'].extend(flipped_matches)

                tile_a['connected_tiles'].append({'view': pair_b, 'tile': tile_b})
                tile_b['connected_tiles'].append({'view': pair_a, 'tile': tile_a})
                
                pair['flipped'] = flipped_matches
        
        return map

    def assign_weights(self, map):
        # TODO - implement for groups/split affine
        pass

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

    def assign_views_to_tiles(self):
        map = {}
        for view in self.view_id_set:
            tp, setup = view
            key = f"timepoint: {tp}, setup: {setup}"

            map[key] = {
                'view': key,
                'connected_tiles': [],
                'cost': 0,
                'distance': 0,
                'matches': [],
                'model': self.create_models()
            }
        
        return map

    # compute tiles
    def init_global_opt(self):
        # TODO - presteps for grouping / split affine
        group = None

        map = self.assign_views_to_tiles()
        weighted_map = self.assign_weights(map) # TODO - for grouping / split affine
        map = self.assign_point_matches(map)

        return map, group
    
    def is_contained(self, key, group):
        if group == None:
            return None
        
        # TODO -implement group / split affine handling
    
    def add_and_fix_tiles(self, map, group):
        tc = {
            'error': 0,
            'fixed_tiles': [],
            'max_error': 0,
            'min_error': float('inf'),
            'tiles': []
        }

        tiles = []
        for tp, setup in self.view_id_set:
            key = f"timepoint: {tp}, setup: {setup}"
            tile = map[key]
            
            # if key in self.fixed_views and tile not in tc['fixed_tiles']:
            #     fixed_group = self.is_contained(key, group)  # For grouping / split affine 
            #     tc['fixed_tiles'].append(tile)
            
            tiles.append(tile)
        
        for tile in tiles:
            if len(tile['connected_tiles']) > 0:
            # if len(tile['connected_tiles']) > 0 or tile in tc['fixed_tiles']:
                tc['tiles'].append(tile)
        
        return tc
    
    # use tiles to do global optimization
    def compute_tiles(self):
        map, group = self.init_global_opt()
        tc = self.add_and_fix_tiles(map, group)

        if len(tc['tiles']) == 0:
            return None
        else:
            return tc

    def run(self):
        """
        Executes the entry point of the script.
        """
        tc = self.compute_tiles()
        return tc