import zarr
import s3fs

class DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.store = self._initialize_store()

    def _initialize_store(self):
        if self.base_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            return s3fs.S3Map(root=self.base_path, s3=s3, check=False)
        else:
            return zarr.N5Store(self.base_path)

    def load_dataset(self, dataset_path):
        root = zarr.open(self.store, mode='r')
        if dataset_path not in root:
            raise KeyError(f"Dataset path {dataset_path} not found.")
        return root[dataset_path]

    def load_interest_points(self, dataset_path):
        loc_path = f"{dataset_path}/interestpoints/loc"
        dataset = self.load_dataset(loc_path)
        return dataset[:]
