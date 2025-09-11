
"""
Connected Graphs splits a tile set into connected components using each tiles connected_tiles links (split-affine).
"""

class ConnectedGraphs:
    def __init__(self, tiles, dataframes):
        self.tiles = tiles
        self.view_registrations = dataframes['view_registrations']

    def identify_connected_graphs(self):
        """
        Build an undirected graph of tiles
        """
        view_to_tile = {t['view']: t for t in self.tiles}

        # undirected adjacency by view_id
        adj = {v: set() for v in view_to_tile}
        for t in self.tiles:
            v = t['view']
            for conn in t.get('connected_tiles', []):
                nv = conn['view'] if isinstance(conn, dict) else conn
                if nv in view_to_tile and nv != v:
                    adj[v].add(nv)
                    adj[nv].add(v)

        graphs, visited = [], set()
        for v in adj:
            if v in visited:
                continue
            stack = [v]
            comp_views = set([v])
            visited.add(v)
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if w not in visited:
                        visited.add(w)
                        comp_views.add(w)
                        stack.append(w)

            comp = sorted(comp_views) 
            graphs.append([view_to_tile[x] for x in comp])

        return graphs

    def assemble_views(self, connected):
        """
        Normalize a collection of connected tiles to a set of view IDs
        """
        if connected and isinstance(next(iter(connected)), dict):
            connected_view_ids = {t['view'] for t in connected}
        else:
            connected_view_ids = set(connected)

        group_views = [t['view'] for t in self.tiles if t['view'] in connected_view_ids]

        return {'views': group_views}

    def create_wlpmc(self):
        """
        Initialize weak link point match correspondences
        """
        views = [t['view'] for t in self.tiles]
        return {
            'view': views,
            'models': self.tiles,
            'overlap_detection': None,
            'view_registrations': self.view_registrations
        }
    
    def label_subsets(self, group):
        """
        Sort a group's view IDs by setup, then collapse consecutive setup numbers into ranges
        """
        vs = sorted(group['views'], key=lambda v: int(v.partition('setup:')[2].split()[0]))
        tp = int(vs[0].partition('timepoint:')[2].split(',')[0]) if vs else 0

        nums = sorted({int(v.partition('setup:')[2].split()[0]) for v in vs})
        labels, starts = [], []
        if nums:
            s = e = nums[0]
            for x in nums[1:]:
                if x == e + 1:
                    e = x
                else:
                    labels.append(f"{tp}-{s} >-> {tp}-{e}"); starts.append(s)
                    s = e = x
            labels.append(f"{tp}-{s} >-> {tp}-{e}"); starts.append(s)

        group['views'] = vs
        group['subset_labels'] = labels
        return group

    def run(self):
        """
        Executes the entry point of the script.
        """
        graph_sets = self.identify_connected_graphs()
        groups_new = []

        if len(graph_sets) == 1:
            return self.tiles, groups_new
        else:
            for connected in graph_sets:
                group = self.assemble_views(connected)
                group = self.label_subsets(group)
                groups_new.append(group)
        
        wlpmc = self.create_wlpmc()
        return wlpmc, groups_new