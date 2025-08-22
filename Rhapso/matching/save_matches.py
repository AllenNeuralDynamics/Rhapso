import zarr
from collections import defaultdict
import s3fs

"""
Utility class for saving (matched) corresponding interest points to N5 format
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
        grouped_by_viewA = defaultdict(list)

        for idxA, _, viewA, idxB, _, viewB in self.all_results:  
            grouped_by_viewA[viewA].append((idxA, idxB, viewB))
            grouped_by_viewA[viewB].append((idxB, idxA, viewA))  

        idMap = {}
        view_keys = set()
        label = matched_views[0][2]
        for viewA, matches in grouped_by_viewA.items():
            tpA = int(viewA.split("tpId=")[1].split(",")[0])
            vsA = int(viewA.split("setupId=")[1].split(")")[0])
            keyA = f"{tpA},{vsA},{label}"
            view_keys.add(keyA)

            for _, _, viewB in matches:
                tpB = int(viewB.split("tpId=")[1].split(",")[0])
                vsB = int(viewB.split("setupId=")[1].split(")")[0])

                if tpB == tpA:
                    keyB = f"{tpB},{vsB},{label}"
                    view_keys.add(keyB)

        idMap = {key: i for i, key in enumerate(sorted(view_keys))}
        grouped_with_ids = defaultdict(list)

        for viewA, matches in grouped_by_viewA.items():
            for idxA, idxB, viewB in matches:
                tpB = int(viewB.split("tpId=")[1].split(",")[0])
                vsB = int(viewB.split("setupId=")[1].split(")")[0])
                key = f"{tpB},{vsB},{label}"
                view_id = idMap[key]
                grouped_with_ids[viewA].append((idxA, idxB, view_id))

        for viewA, corr_list in grouped_with_ids.items():
            tpA = int(viewA.split("tpId=")[1].split(",")[0])
            vsA = int(viewA.split("setupId=")[1].split(")")[0])

            # Get label from matched_views
            labelA = 'beads'
            for tp, vs, label in matched_views:
                if tp == tpA and vs == vsA:
                    labelA = label
                    break

            if len(corr_list) == 0:
                continue

            # Output path
            full_path = f"{self.n5_output_path}interestpoints.n5/tpId_{tpA}_viewSetupId_{vsA}/{labelA}/correspondences/"

            if full_path.startswith("s3://"):
                path = full_path.replace("s3://", "")
                self.s3_filesystem = s3fs.S3FileSystem()
                store = s3fs.S3Map(root=path, s3=self.s3_filesystem, check=False) 
                root = zarr.open_group(store=store, mode='a')
            else:
                # Write to Zarr N5
                store = zarr.N5Store(full_path)
                root = zarr.group(store=store, overwrite="true")

            # Delete existing 'data' array
            if "data" in root:
                del root["data"]

            # Set group-level attributes
            root.attrs.update({
                "correspondences": "1.0.0",
                "idMap": idMap
            })

            # Create dataset inside the group
            root.create_dataset(
                name="data",  # just the dataset name, not a full path
                data=corr_list,
                dtype='u8',
                chunks=(min(300000, len(corr_list)), 1),
                compressor=zarr.GZip()
            )

    def clear_correspondence(self):
        if self.n5_output_path.startswith("s3://"):
            root_path = self.n5_output_path.replace("s3://", "") + "interestpoints.n5"
            s3 = s3fs.S3FileSystem()
            store = s3fs.S3Map(root=root_path, s3=s3, check=False)
        else:
            root_path = self.n5_output_path + "interestpoints.n5"
            store = zarr.N5Store(root_path)

        root = zarr.open_group(store=store, mode="a")

        views = list(self.data_global['viewsInterestPoints'].keys())  
        for tp, vs in views:
            corr_path = f"tpId_{tp}_viewSetupId_{vs}/beads/correspondences"
            try:
                if corr_path in root:
                    del root[corr_path]                
                elif f"{corr_path}/data" in root:
                    del root[f"{corr_path}/data"]       
            except Exception as e:
                print(f"⚠️ Could not delete {corr_path}: {e}")

    def run(self):
        self.clear_correspondence()
        # Gather all unique (timepoint, setup, label) from the dataset
        views_interest_points = self.data_global['viewsInterestPoints']
        matched_views = []
        for (tp, setup), view_info in views_interest_points.items():
            label = view_info.get('label', 'beads')
            matched_views.append((int(tp), int(setup), label))

        # Save correspondences
        self.save_correspondences(matched_views)