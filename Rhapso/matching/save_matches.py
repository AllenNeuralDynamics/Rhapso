import zarr
import numpy as np
from collections import defaultdict
import s3fs

"""
Save Matches saves matched corresponding interest points to N5 format.
"""

class SaveMatches:
    def __init__(self, all_results, n5_output_path, data_global, match_type):
        self.all_results = all_results
        self.n5_output_path = n5_output_path
        self.data_global = data_global
        self.match_type = match_type

    def write_one_block_dataset(self, root, name, data, dtype, attrs):
        """
        Write matches as one block/chunk.
        """
        data = np.asarray(data, dtype=dtype)

        # Empty datasets can have shape 0, but chunk dimensions cannot be 0.
        chunks = tuple(max(1, dim) for dim in data.shape)

        if name in root:
            arr = zarr.creation.create(
                shape=data.shape,
                chunks=chunks,
                dtype=dtype,
                compressor=zarr.GZip(),
                store=root.store,
                path=f"{root.path}/{name}" if root.path else name,
                overwrite=True,
            )

            if data.size > 0:
                arr[...] = data

        else:
            arr = root.create_dataset(
                name=name,
                data=data,
                dtype=dtype,
                chunks=chunks,
                compressor=zarr.GZip(),
            )

        for k, v in attrs.items():
            arr.attrs[k] = v

        return arr

    def parse_view(self, view: str):
        tp = int(view.split("tpId=")[1].split(",")[0])
        vs = int(view.split("setupId=")[1].split(")")[0])
        return tp, vs

    def format_view(self, tp, vs):
        return f"(tpId={tp}, setupId={vs})"

    def open_correspondence_group(self, tp, vs, label):
        full_path = (
            f"{self.n5_output_path}"
            f"interestpoints.n5/tpId_{tp}_viewSetupId_{vs}/{label}/correspondences/"
        )

        if full_path.startswith("s3://"):
            path = full_path.replace("s3://", "", 1)
            s3_filesystem = s3fs.S3FileSystem()
            store = s3fs.S3Map(root=path, s3=s3_filesystem, check=False)
            root = zarr.open_group(store=store, mode="a")
        else:
            store = zarr.N5Store(full_path)
            root = zarr.open_group(store=store, mode="a")

        return root

    def save_correspondences(self):
        """
        Save correspondences for every expected view/label.
        """
        # Group results per source view.
        grouped_by_view = defaultdict(list)

        for idxA, _, viewA, labelA, idxB, _, viewB, labelB in self.all_results:
            grouped_by_view[viewA].append((idxA, idxB, viewB, labelB))
            grouped_by_view[viewB].append((idxB, idxA, viewA, labelA))

        # Build idMap per source view.
        idMaps = {}

        for viewA, matches in grouped_by_view.items():
            target_keys = sorted({
                f"{tpB},{vsB},{labelB}"
                for (_idxA, _idxB, viewB, labelB) in matches
                for (tpB, vsB) in [self.parse_view(viewB)]
            })

            idMaps[viewA] = {key: i for i, key in enumerate(target_keys)}

        # Convert grouped matches into rows: idxA, idxB, target_view_id.
        grouped_with_ids = defaultdict(list)

        for viewA, matches in grouped_by_view.items():
            idMap = idMaps[viewA]

            for idxA, idxB, viewB, labelB in matches:
                tpB, vsB = self.parse_view(viewB)
                key = f"{tpB},{vsB},{labelB}"
                target_view_id = idMap[key]

                # Keep correspondences grouped by source view and source label.
                # labelB is the label of the target view, but in this pipeline labels are usually "beads".
                grouped_with_ids[(viewA, labelB)].append((idxA, idxB, target_view_id))

        # Loop over every expected view/label so stale old matches get overwritten with empty data.
        for tp, vs in self.data_global["viewsInterestPoints"].keys():
            labels = self.data_global["viewsInterestPoints"][(tp, vs)]["label"]
            viewA = self.format_view(tp, vs)

            for label in labels:
                corr_list = grouped_with_ids.get((viewA, label), [])
                idMap = idMaps.get(viewA, {})

                root = self.open_correspondence_group(tp, vs, label)

                # Rewrite group metadata.
                root.attrs.update({
                    "correspondences": "1.0.0",
                    "idMap": idMap,
                })

                if len(corr_list) > 0:
                    corr_data = np.asarray(corr_list, dtype=np.uint64).reshape(-1, 3)
                else:
                    corr_data = np.empty((0, 3), dtype=np.uint64)

                num_corr = corr_data.shape[0]

                self.write_one_block_dataset(
                    root=root,
                    name="data",
                    data=corr_data,
                    dtype="u8",
                    attrs={
                        "dimensions": [num_corr, 3],
                        "blockSize": [max(num_corr, 1), 3],
                    },
                )

    def run(self):
        self.save_correspondences()