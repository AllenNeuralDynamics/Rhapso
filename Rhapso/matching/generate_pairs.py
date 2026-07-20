from itertools import combinations
import numpy as np

"""
Generate match pairs
"""

class GeneratePairs:
    def __init__(self, data_global, match_type):
        self.data_global = data_global
        self.match_type = match_type
    
    def set_id(self, v1, v_sets):
        """
        Find the index of the component in `v_sets` that contains `v1`
        """
        i = -1
        for j in range(len(v_sets)):
            if v1 in v_sets[j]:
                i = j
        
        return i
    
    def subsets(self, pairs):
        """
        Cluster views into connected components based on the given pairs
        """
        views = list(self.data_global['viewsInterestPoints'].keys())
        v_sets: list[set] = []
        pair_sets: list[list[tuple]] = []       
        groups = None

        counter = 0

        for pair_a, pair_b in pairs:
            
            counter += 1
            if counter == 100:
                break
            
            i1 = self.set_id(pair_a, v_sets)
            i2 = self.set_id(pair_b, v_sets)

            if i1 == -1 and i2 == -1:
                v_set: list[set] = []
                pair_set: list[set] = []
                pair_set.append((pair_a, pair_b))
                v_set.append(pair_a)
                v_set.append(pair_b)

                v_sets.append(v_set)
                pair_sets.append(pair_set)
            
            elif i1 >= 0 and i2 == 0:
                v_sets[i2].append(pair_a)
                pair_sets[i2].append((pair_a, pair_b))
            
            elif i1 >= 0 and i2 == -1:
                v_sets[i1].append(pair_b)
                pair_sets[i1].append((pair_a, pair_b))
            
            elif i1 == i2:
                pair_sets[i1].append((pair_a, pair_b))
            
            else:
                pair_sets, v_sets = self.merge_sets(v_sets, pair_sets, i1, i2)
        
        for view in views:
            is_present = False

            for subset_precursor in v_sets:
                if view in subset_precursor:
                    is_present = True

            if not is_present:
                v_set = []
                pair_set = []

                v_set.append(view)
                v_sets.append(v_set)
                pair_sets.append(pair_set)

        subsets = []

        for i in range(len(v_sets)):
            set_pairs = pair_sets[i]
            set_views = v_sets[i]
            subsets.append((set_views, set_pairs, groups))
        
        return {
            'groups': None,
            'pairs': pairs,
            'rangeComparator': None,
            'subsets': subsets,
            'views': views 
        }
    
    def get_bounding_boxes(self, matrix, dims):
        """
        Compute the world-space axis-aligned bounding box of a voxel-aligned view.
        """
        matrix = np.asarray(matrix, dtype=float)

        if matrix.shape == (3, 4):
            matrix = np.vstack([
                matrix,
                [0.0, 0.0, 0.0, 1.0],
            ])

        size_x = float(dims[0]) - 1.0
        size_y = float(dims[1]) - 1.0
        size_z = float(dims[2]) - 1.0

        linear = matrix[:3, :3]
        translation = matrix[:3, 3]

        minimum = translation.copy()
        maximum = translation.copy()

        sizes = (size_x, size_y, size_z)

        for output_axis in range(3):
            for input_axis, size in enumerate(sizes):
                contribution = size * linear[output_axis, input_axis]

                if contribution < 0:
                    minimum[output_axis] += contribution
                else:
                    maximum[output_axis] += contribution

        return minimum, maximum

    def bounding_boxes(self, matrix, dims):
        """
        Convert the real-valued bounds to a padded integer bounding box.
        """
        minimum, maximum = self.get_bounding_boxes(
            matrix,
            dims["size"],
        )

        minimum_integer = np.rint(minimum).astype(int) - 1
        maximum_integer = np.rint(maximum).astype(int) + 1

        return minimum_integer.tolist(), maximum_integer.tolist()


    def transform_matrices(self, view):
        """
        Compose all registered transforms for a view.
        """
        matrix = np.eye(4, dtype=float)

        for model in self.data_global["viewRegistrations"].get(view, []):
            values = np.fromstring(
                str(model["affine"]).replace(",", " "),
                sep=" ",
                dtype=float,
            )

            if values.size != 12:
                raise ValueError(
                    f"Expected 12 affine values for view {view}, "
                    f"got {values.size}"
                )

            transform = np.eye(4, dtype=float)
            transform[:3, :4] = values.reshape(3, 4)

            matrix = matrix @ transform

        return matrix


    def overlaps(self, box_a, box_b):
        """
        Return True when two axis-aligned bounding boxes intersect.
        """
        minimum_a, maximum_a = box_a
        minimum_b, maximum_b = box_b

        for axis in range(3):
            if maximum_a[axis] < minimum_b[axis]:
                return False

            if maximum_b[axis] < minimum_a[axis]:
                return False

        return True


    def overlap(self, view_a, dims_a, view_b, dims_b):
        """
        Determine whether two transformed views overlap.
        """
        matrix_a = self.transform_matrices(view_a)
        matrix_b = self.transform_matrices(view_b)

        box_a = self.bounding_boxes(matrix_a, dims_a)
        box_b = self.bounding_boxes(matrix_b, dims_b)

        return self.overlaps(box_a, box_b)
    
    def setup_groups_split(self):
        """
        Generate all unique view pairs and keep only those whose setups overlap
        """
        views = list(self.data_global['viewsInterestPoints'].keys())
        pairs = list(combinations(views, 2))
        final_pairs = []

        for view_a, view_b in pairs:    
            dims_a = self.data_global['viewSetup']['byId'][view_a[1]]
            dims_b = self.data_global['viewSetup']['byId'][view_b[1]]
            
            if self.overlap(view_a, dims_a, view_b, dims_b):
                view_a = (view_a[0], view_a[1])
                view_b = (view_b[0], view_b[1])
                final_pairs.append((view_a, view_b))
        
        return final_pairs
    
    def is_face_neighbor(self, view_a, dims_a, view_b, dims_b, min_shared_fraction=0.5):
        matrix_a = self.transform_matrices(view_a)
        matrix_b = self.transform_matrices(view_b)

        minimum_a, maximum_a = self.get_bounding_boxes(matrix_a, dims_a["size"])
        minimum_b, maximum_b = self.get_bounding_boxes(matrix_b, dims_b["size"])

        extent_a = maximum_a - minimum_a
        extent_b = maximum_b - minimum_b
        shared = np.minimum(maximum_a, maximum_b) - np.maximum(minimum_a, minimum_b)

        if np.any(shared <= 0):
            return False

        shared_fraction = shared / np.maximum(np.minimum(extent_a, extent_b), 1e-9)
        return np.count_nonzero(shared_fraction >= min_shared_fraction) >= 2
    
    def setup_groups(self):
        """
        Group views by timepoint and generate face-neighbor pairs only.
        """
        views = list(self.data_global["viewsInterestPoints"].keys())
        timepoint_groups = {}

        for view in views:
            timepoint_groups.setdefault(view[0], []).append(view)

        pairs = []

        for timepoint_views in timepoint_groups.values():
            for view_a, view_b in combinations(timepoint_views, 2):
                dims_a = self.data_global["viewSetup"]["byId"][view_a[1]]
                dims_b = self.data_global["viewSetup"]["byId"][view_b[1]]

                if self.is_face_neighbor(view_a, dims_a, view_b, dims_b):
                    pairs.append((view_a, view_b))

        return {
            "groups": timepoint_groups,
            "pairs": pairs,
            "rangeComparator": None,
            "subsets": None,
            "views": views,
        }
    
    # def setup_groups(self):
    #     """
    #     Group views by timepoint and generate all unique unordered intra-timepoint pairs
    #     """
    #     views = list(self.data_global['viewsInterestPoints'].keys())

    #     # Group views by timepoint
    #     timepoint_groups = {}
    #     for view in views:
    #         timepoint, _ = view
    #         if timepoint not in timepoint_groups:
    #             timepoint_groups[timepoint] = []
    #         timepoint_groups[timepoint].append(view)

    #     # Create pairs within each timepoint
    #     pairs = []
    #     for timepoint, timepoint_views in timepoint_groups.items():
    #         for i in range(len(timepoint_views)):
    #             for j in range(i + 1, len(timepoint_views)):
    #                 pairs.append((timepoint_views[i], timepoint_views[j]))

    #     return {
    #         'groups': timepoint_groups,
    #         'pairs': pairs,
    #         'rangeComparator': None,
    #         'subsets': None,
    #         'views': views
    #     }
    
    def as_list(self, x):
        return x if isinstance(x, list) else [x]

    def expand_pairs_with_labels(self, base_pairs, view_ids_global):
        """
        Add a label for each pair
        """
        out = []
        for va, vb in base_pairs:
            la = self.as_list(view_ids_global[va].get('label', []))
            lb = self.as_list(view_ids_global[vb].get('label', []))

            if not la or not lb:
                continue

            lb_set = set(lb)
            common = [l for l in la if l in lb_set]

            for l in common:
                out.append(((va[0], va[1]), (vb[0], vb[1]), l))

        return out

    def generate_pairs(self):
        """
        Executes the entry point of the script.
        """
        view_ids_global = self.data_global['viewsInterestPoints']
        view_registrations = self.data_global['viewRegistrations']

        # Set up view groups using complete dataset info
        if self.match_type == "split-affine":
            setup = self.setup_groups_split()
            setup = self.subsets(setup)
        else:
            setup = self.setup_groups()

        setup['pairs'] = self.expand_pairs_with_labels(setup['pairs'], view_ids_global)
        
        process_pairs = []
        for view_a, view_b, label in setup['pairs']:
            if isinstance(view_a, tuple) and len(view_a) == 2:
                tpA, setupA = view_a
                viewA_str = f"(tpId={tpA}, setupId={setupA})"
            else:
                viewA_str = str(view_a)

            if isinstance(view_b, tuple) and len(view_b) == 2:
                tpB, setupB = view_b
                viewB_str = f"(tpId={tpB}, setupId={setupB})"
            else:
                viewB_str = str(view_b)

            process_pairs.append((view_a, view_b, viewA_str, viewB_str, label))

        return process_pairs, view_registrations

    def run(self):
        print("Match Pairs Generated")

        return self.generate_pairs()
        