import zarr
from collections import defaultdict
import s3fs

"""
Save Matches saves (matched) corresponding interest points to N5 format
"""

class SaveMatches:
    def __init__(self, all_results, n5_output_path, data_global, match_type):
        self.all_results = all_results
        self.n5_output_path = n5_output_path
        self.data_global = data_global
        self.match_type = match_type
    
    def save_correspondences(self):
        """
        Save correspondences for each view/label, aggregating all matches involving that view/label.
        Print a detailed summary with breakdowns.
        """
        def parse_view(v: str):
            tp = int(v.split("tpId=")[1].split(",")[0])
            vs = int(v.split("setupId=")[1].split(")")[0])
            return tp, vs

        # Group results back per view
        grouped_by_viewA = defaultdict(list)
        for idxA, _, viewA, label_a, idxB, _, viewB, label_b in self.all_results:  
            grouped_by_viewA[viewA].append((idxA, idxB, viewB, label_b))
            grouped_by_viewA[viewB].append((idxB, idxA, viewA, label_a)) 

        # Create idmap per view of all corresponding groups
        idMaps = {}
        for viewA, matches in grouped_by_viewA.items():
            target_keys = sorted({
                f"{tpB},{vsB},{labB}"
                for (_iA, _iB, viewB, labB) in matches
                for (tpB, vsB) in [parse_view(viewB)]
            })
            idMaps[viewA] = {k: i for i, k in enumerate(target_keys)}

        # Format data for injection
        grouped_with_ids = defaultdict(list)
        for viewA, matches in grouped_by_viewA.items():
            idMap = idMaps[viewA]
            for idxA, idxB, viewB, label in matches:
                tp = int(viewB.split("tpId=")[1].split(",")[0])
                vs = int(viewB.split("setupId=")[1].split(")")[0])
                key = f"{tp},{vs},{label}"
                view_id = idMap[key]
                grouped_with_ids[viewA, label].append((idxA, idxB, view_id))

        # Save idmap and corr points per view
        for (viewA, labelA), corr_list in grouped_with_ids.items():
            tpA = int(viewA.split("tpId=")[1].split(",")[0])
            vsA = int(viewA.split("setupId=")[1].split(")")[0])
            idMap = idMaps[viewA]

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
                name="data",  
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
            labels = self.data_global['viewsInterestPoints'][(tp, vs)]['label']
            for label in labels:
                corr_path = f"tpId_{tp}_viewSetupId_{vs}/{label}/correspondences"
                try:
                    if corr_path in root:
                        del root[corr_path]                
                    elif f"{corr_path}/data" in root:
                        del root[f"{corr_path}/data"]       
                except Exception as e:
                    print(f"⚠️ Could not delete {corr_path}: {e}")

    def run(self):
        self.clear_correspondence()
        self.save_correspondences()