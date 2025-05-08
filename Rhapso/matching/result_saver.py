import zarr
import numpy as np

class ResultSaver:
    def __init__(self, base_path):
        self.base_path = base_path
        self.store = self._initialize_store()

    def _initialize_store(self):
        if self.base_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            return s3fs.S3Map(root=self.base_path, s3=s3, check=False)
        else:
            return zarr.N5Store(self.base_path)

    def save_matches(self, matches, view_paths, clear_correspondences=False):
        root = zarr.group(store=self.store, overwrite=False)
        for match in matches:
            viewA, viewB, idxA, idxB = match
            timepoint, _ = viewA
            group_path = f"tpId_{timepoint}_viewSetupId_{viewA[1]}/beads"
            if group_path not in root:
                group = root.create_group(group_path)
            else:
                group = root[group_path]
            if 'correspondences' in group and clear_correspondences:
                del group['correspondences']
            correspondences = group.require_group('correspondences')
            matches_array = np.array([[idxA, idxB]], dtype=np.uint64)
            correspondences.create_dataset("data", data=matches_array, dtype='uint64', chunks=(1000,))
