from copy import deepcopy

"""
Concatenate Models stitches together the results of two alignment rounds (split-affine)
"""

class ConcatenateModels:
    def __init__(self, tiles, tiles_round_2, groups, validation_stats, validation_stats_round_2, view_map):
        self.tiles = tiles
        self.tiles_round_2 = tiles_round_2
        self.groups = groups
        self.validation_stats = validation_stats
        self.validation_stats_round_2 = validation_stats_round_2
        self.view_map = view_map

    def map_models_back_from_groups(self):
        """
        Copy solved model and matches back onto every tile in the same group
        """
        view_to_group = {v: gi for gi, g in enumerate(self.groups) for v in g.get('views', [])}

        for rep_tile in self.tiles_round_2:
            rep_view = rep_tile['view']
            gi = view_to_group.get(rep_view)
            if gi is None:
                continue  

            group_views = self.groups[gi]['views']
            rep_model  = deepcopy(rep_tile.get('model', {}))
            rep_matches = list(rep_tile.get('matches', []))  

            # Normalize rep connections into dict
            rep_conns = []
            for c in rep_tile.get('connected_tiles', []):
                v = c['view'] if isinstance(c, dict) else c
                rep_conns.append({'view': v, 'tile': self.view_map.get(v)})

            # Propagate to every member of the group
            for v in group_views:
                t = self.view_map.get(v)
                if not t:
                    continue

                t['model']   = deepcopy(rep_model)     
                t['matches'] = list(rep_matches)

        return list(self.view_map.values())
    
    def preconcatenate_affine(self, m1, m2):
        """
        Compose two 3x4 affine transforms stored as dicts
        """
        m00,m01,m02,m03 = m1['m00'],m1['m01'],m1['m02'],m1['m03']
        m10,m11,m12,m13 = m1['m10'],m1['m11'],m1['m12'],m1['m13']
        m20,m21,m22,m23 = m1['m20'],m1['m21'],m1['m22'],m1['m23']

        n00,n01,n02,n03 = m2['m00'],m2['m01'],m2['m02'],m2['m03']
        n10,n11,n12,n13 = m2['m10'],m2['m11'],m2['m12'],m2['m13']
        n20,n21,n22,n23 = m2['m20'],m2['m21'],m2['m22'],m2['m23']

        return {
            'm00': n00*m00 + n01*m10 + n02*m20,
            'm01': n00*m01 + n01*m11 + n02*m21,
            'm02': n00*m02 + n01*m12 + n02*m22,
            'm03': n00*m03 + n01*m13 + n02*m23 + n03,

            'm10': n10*m00 + n11*m10 + n12*m20,
            'm11': n10*m01 + n11*m11 + n12*m21,
            'm12': n10*m02 + n11*m12 + n12*m22,
            'm13': n10*m03 + n11*m13 + n12*m23 + n13,

            'm20': n20*m00 + n21*m10 + n22*m20,
            'm21': n20*m01 + n21*m11 + n22*m21,
            'm22': n20*m02 + n21*m12 + n22*m22,
            'm23': n20*m03 + n21*m13 + n22*m23 + n23,
        }
    
    def merge_validation_stats(self, v1, v2):
        """
        Merge two validation-metrics dicts by concatenating per-tile from v2 after v1, offsetting v2's iteration numbers 
        and tagging entries
        """
        out = deepcopy(v1) if v1 else {}
        s1 = out.setdefault('solver_metrics_per_tile', {}).setdefault('stats', [])
        s2 = (v2 or {}).get('solver_metrics_per_tile', {}).get('stats', []) or []
        
        # tag round and offset iterations
        if s1 and 'round' not in s1[0]:
            for x in s1: x['round'] = 1
        offset = (s1[-1]['iteration'] + 1) if s1 else 0
        
        for x in s2:
            y = dict(x)
            y['iteration'] = x.get('iteration', 0) + offset
            y['round'] = 2
            s1.append(y)
            
        return out

    def run(self):
        """
        Executes the entry point of the script.
        """
        view_map = self.map_models_back_from_groups()
        combined_validation_stats = self.merge_validation_stats(self.validation_stats, self.validation_stats_round_2)

        tiles_round_1 = {t['view']: t for t in self.tiles}
        tiles_round_2  = {t['view']: t for t in view_map}

        for vid in tiles_round_1.keys() & tiles_round_2.keys():
            m1 = tiles_round_1[vid]['model']['regularized']
            m2 = tiles_round_2[vid]['model']['regularized']
            tiles_round_2[vid]['model']['regularized'] = self.preconcatenate_affine(m1, m2) 
        
        tiles_round_2 = [tiles_round_2[t['view']] for t in view_map]
        return tiles_round_2, combined_validation_stats