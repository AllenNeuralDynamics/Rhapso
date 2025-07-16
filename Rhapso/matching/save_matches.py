import numpy as np
import zarr
from collections import defaultdict

"""
Functions for saving data to N5 format, specifically designed for interest points and correspondences.
Uses zarr instead of z5py for better compatibility.
"""

class SaveMatches:
    def __init__(self, all_results, n5_output_path, data_global):
        self.all_results = all_results
        self.n5_output_path = n5_output_path
        self.data_global = data_global

    def save_correspondences(self, matched_views):
        """
        Save correspondences for each view/label, aggregating all matches involving that view/label.
        Print a detailed summary with breakdowns.
        """
        # (tp, vs, label) -> [(other_tp, other_vs, other_label, idx_self, idx_other)]
        per_view_corrs = defaultdict(list)
        pair_breakdown = defaultdict(lambda: defaultdict(list))

        for idxA, cordsA, viewA, idxB, cordsB, viewB in self.all_results:
            def get_label(tp, vs):
                for v in matched_views:
                    if int(v[0]) == int(tp) and int(v[1]) == int(vs):
                        return v[2]
                return 'beads'

            tpA = viewA.split("tpId=")[1].split(",")[0]
            vsA = viewA.split("setupId=")[1].split(")")[0]
            tpB = viewB.split("tpId=")[1].split(",")[0]
            vsB = viewB.split("setupId=")[1].split(")")[0]

            labelA = get_label(tpA, vsA)
            labelB = get_label(tpB, vsB)

            keyA = (int(tpA), int(vsA), labelA)
            keyB = (int(tpB), int(vsB), labelB)

            per_view_corrs[keyA].append((keyB[0], keyB[1], keyB[2], idxA, idxB))
            per_view_corrs[keyB].append((keyA[0], keyA[1], keyA[2], idxB, idxA))

            pair_breakdown[keyA][keyB].append((idxA, idxB))
            pair_breakdown[keyB][keyA].append((idxB, idxA))

        for idx, view in enumerate(matched_views, 1):
            tp, vs, label = int(view[0]), int(view[1]), view[2]
            key = (tp, vs, label)
            corr_list = per_view_corrs.get(key, [])

            if not corr_list:
                continue

            # Build idMap from (tp, vs, label) -> id
            unique_keys = {
                f"{tp_b},{vs_b},{label_b}" for tp_b, vs_b, label_b, _, _ in corr_list
                if (tp_b, vs_b, label_b) != key
            }
            idMap = {k: i for i, k in enumerate(sorted(unique_keys))}

            # Convert to compact format: [idx_self, idx_other, idMap[other_view]]
            compact_matches = []
            for tp_b, vs_b, label_b, idx_self, idx_other in corr_list:
                view_str = f"{tp_b},{vs_b},{label_b}"
                if view_str not in idMap:
                    continue
                compact_matches.append([idx_self, idx_other, idMap[view_str]])

            compact_array = np.array(compact_matches, dtype=np.uint64)

            print(f"Saving {len(compact_array)} matches for view tpId={tp}, setupId={vs}, label={label}")

            full_path = self.n5_output_path + f"/interestpoints.n5/tpId_{tp}_viewSetupId_{vs}/{label}/correspondences"

            # Open Zarr N5 store
            store = zarr.N5Store(full_path)
            root = zarr.group(store=store)

            # Add attributes to correspondences/
            root.attrs.update({
                "correspondences": "1.0.0",
                "idMap": idMap
            })

            if "data" in root:
                del root["data"]  

            root.create_dataset(
                "data",
                data=compact_array,
                dtype='u8',
                chunks=(min(300000, len(compact_array)), 3),
                compressor=zarr.GZip()
            )

    def run(self):
        print("Gathering all unique (timepoint, setup, label) from the dataset...")
        views_interest_points = self.data_global['viewsInterestPoints']
        matched_views = []
        for (tp, setup), view_info in views_interest_points.items():
            label = view_info.get('label', 'beads')
            matched_views.append((int(tp), int(setup), label))
        print(f"Found {len(matched_views)} unique views with interest points.")

        print("Saving correspondences for each view...")
        self.save_correspondences(matched_views)
        print("Correspondences saved successfully.")