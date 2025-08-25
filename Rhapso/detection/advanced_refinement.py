from collections import OrderedDict
from scipy.spatial import cKDTree
import numpy as np
from collections import defaultdict, OrderedDict

"""
Filter interest points per bound and remove duplicates
"""

class AdvancedRefinement:
    def __init__(self, interest_points, combine_distance, dataframes, overlapping_area, max_interval_size, max_spots):
        self.interest_points = interest_points
        self.consolidated_data = {}
        self.combine_distance = combine_distance
        self.image_loader_df = dataframes['image_loader']
        self.overlapping_area = overlapping_area
        self.max_interval_size = max_interval_size
        self.max_spots = max_spots
        self.overlapping_only = True
        self.sorted_view_ids = None
        self.result = interest_points  
        self.store_intensities = False
        self._max_spots = 0
        self.max_spots_per_overlap = False
        self.to_process = interest_points
        self.interest_points_per_view_id = {}
        self.intensities_per_view_id = {}
        self.intervals_per_view_id = {}

    def kd_tree(self, ips_lists_by_view, ints_lists_by_view):
        radius = float(self.combine_distance)
        out = OrderedDict()

        for view_id in sorted(ips_lists_by_view.keys()):
            ips_lists = ips_lists_by_view[view_id]
            ints_lists = ints_lists_by_view[view_id]

            my_ips: list = []
            my_ints: list = []

            for l, ips in enumerate(ips_lists):
                intens = ints_lists[l]

                # First list: accept all 
                if not my_ips:
                    my_ips.extend(ips)
                    my_ints.extend(intens)
                    continue

                # Build KDTree from the CURRENT accepted points for this view
                base = np.asarray(my_ips, dtype=np.float32)
                tree = cKDTree(base)

                # Batch query all new points against the tree
                cand = np.asarray(ips, dtype=np.float32)
                
                if cand.size == 0:
                    continue
                
                dists, _ = tree.query(cand, k=1)  

                # Keep only points farther than combineDistance
                mask = dists > radius
                if np.any(mask):
                    # Extend accepted sets
                    for p, val in zip(cand[mask], np.asarray(intens)[mask]):
                        my_ips.append(p.tolist())   
                        my_ints.append(float(val))

            # Store consolidated (point, intensity) pairs per view
            out[view_id] = list(zip(my_ips, my_ints))

        self.consolidated_data = out
    
    def size(self, interval):
        lb, ub = interval[0], interval[1]
        prod = 1
        for l, u in zip(lb, ub):
            prod *= (int(u) - int(l) + 1)
        return prod
    
    def contains(self, containing, contained):
        lc, uc = containing[0], containing[1]
        li, ui = contained[0],  contained[1]
        return all(lc[d] <= li[d] and uc[d] >= ui[d] for d in range(3))
    
    def filter_lists(self, ips, intensities, my_max_spots):
        if intensities is None or len(ips) == 0 or my_max_spots <= 0:
            return ips, intensities

        intens_arr = np.asarray(intensities)
        n = min(len(ips), intens_arr.shape[0])
        if n == 0:
            return ips, intensities

        # indices of top-N by descending intensity
        top_idx = np.argsort(intens_arr[:n])[::-1][:my_max_spots]

        # slice ips preserving original type
        if isinstance(ips, np.ndarray):
            ips_filtered = ips[top_idx]
        else:
            ips_filtered = [ips[i] for i in top_idx]

        intens_filtered = intens_arr[top_idx]
        if isinstance(intensities, list):
            intens_filtered = intens_filtered.tolist()

        return ips_filtered, intens_filtered
        
    def filter(self):
        ips_lists_by_view = defaultdict(list)
        ints_lists_by_view = defaultdict(list)
        intervals_by_view = defaultdict(list)

        # Prep lists of interest points
        for entry in self.interest_points:
            vid = entry["view_id"]
            ips = entry["interest_points"]      
            intens = entry["intensities"] 
            interval = entry["interval_key"]     
            ips_lists_by_view[vid].append(ips)
            ints_lists_by_view[vid].append(intens)
            intervals_by_view[vid].append(interval)

        for i, row_i in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row_i['timepoint']}, setup: {row_i['view_setup']}"

            ips_list = ips_lists_by_view[view_id]
            intensities_list = ints_lists_by_view[view_id]
            interval_list = intervals_by_view[view_id]

            if not interval_list or not ips_list:
                continue

            interval_data = []

            to_process = [
                {'view_id': vid, **d}
                for vid, lst in self.overlapping_area.items()
                for d in lst
            ]
            
            for row in to_process:
                vid = row['view_id']
                lb = row['lower_bound']
                ub = row['upper_bound']
                if vid == view_id:
                    to_process_interval = (lb, ub)
                    ips_block = []
                    intensities_block = []

                    for i in range(len(ips_list)): 
                        block_interval = interval_list[i]
                        
                        if self.contains(to_process_interval, block_interval):
                            ips_block.extend(ips_list[i])
                            intensities_block.extend(intensities_list[i])
                    
                    interval_data.append((to_process_interval, ips_block, intensities_block))
            
            ips_lists_by_view[view_id] = []
            ints_lists_by_view[view_id] = []

            for interval, ips, intensities in interval_data:
                size = self.size(interval)
                my_max_spots = int(round(self.max_spots * (size / self.max_interval_size)))
                
                if my_max_spots > 0 and my_max_spots < len(ips):
                    ips, intensities = self.filter_lists(ips, intensities, my_max_spots)

                ips_lists_by_view[view_id].append(ips)
                ints_lists_by_view[view_id].append(intensities)
        
        return ips_lists_by_view, ints_lists_by_view

    def run(self):
        """
        Executes the entry point of the script.
        """
        ips_lists_by_view, ints_lits_by_view = self.filter()
        self.kd_tree(ips_lists_by_view, ints_lits_by_view)
        
        return self.consolidated_data