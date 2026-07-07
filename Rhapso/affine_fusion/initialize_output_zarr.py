import numpy as np
import zarr
import os

"""
Initialize output zarr store/group in S3 or local storage.
Works with both Zarr-Python 2 and Zarr-Python 3 APIs.
"""

class InitializeOutputZarr:
    def __init__(self, output_path, zarr_input_prefix, dims, output_zarr_version):
        self.output_path = str(output_path).rstrip("/")
        self.zarr_input_prefix = str(zarr_input_prefix).rstrip("/")
        self.output_shape_zyx = (int(dims[2]), int(dims[1]), int(dims[0]))
        self.output_zarr_version = int(output_zarr_version)

        if self.output_zarr_version not in (2, 3):
            raise ValueError(
                f"output_zarr_version must be 2 or 3, got {self.output_zarr_version}"
            )

        self.has_zarr3_store_api = hasattr(zarr.storage, "FsspecStore")

        if not self.has_zarr3_store_api and self.output_zarr_version == 3:
            raise RuntimeError(
                "This environment has the Zarr 2 Python API, which cannot create "
                "Zarr format 3 output. Install zarr>=3 or set output_zarr_version=2."
            )
        
    def first_child_zarr(self, path, anon=None):
        """
        If path is a folder containing tile zarrs, step into next level 
        for v3 SPIM layout
        """
        path = str(path).rstrip("/")

        if path.startswith("s3://"):
            import fsspec

            fs = fsspec.filesystem(
                "s3",
                anon=bool(anon) if anon is not None else False,
            )
            key = path[len("s3://"):].rstrip("/")
            children = fs.ls(key, detail=False)

            for child in sorted(str(c).rstrip("/") for c in children):
                name = child.rsplit("/", 1)[-1]
                if name.endswith(".ome.zarr") or name.endswith(".zarr"):
                    return child if child.startswith("s3://") else "s3://" + child

            return None

        if not os.path.isdir(path):
            return None

        for name in sorted(os.listdir(path)):
            if name.endswith(".ome.zarr") or name.endswith(".zarr"):
                return os.path.join(path, name)

        return None

    def make_store(self, path, mode="r", anon=None):
        """
        Create a store for local or S3 paths using the installed Zarr API.
        """
        path = str(path).rstrip("/")

        if self.has_zarr3_store_api:
            return self.make_store_zarr3(path, mode=mode, anon=anon)

        return self.make_store_zarr2(path, mode=mode, anon=anon)

    def make_store_zarr3(self, path, mode="r", anon=None):
        """
        Zarr-Python 3 store API.
        """
        read_only = mode == "r"

        if path.startswith("s3://"):
            storage_options = {}
            if anon is not None:
                storage_options["anon"] = bool(anon)

            return zarr.storage.FsspecStore.from_url(
                path,
                storage_options=storage_options,
                read_only=read_only,
            )

        return zarr.storage.LocalStore(
            path,
            read_only=read_only,
        )

    def make_store_zarr2(self, path, mode="r", anon=None):
        """
        Zarr-Python 2 store API.
        """
        path = str(path).rstrip("/")

        if path.startswith("s3://"):
            storage_options = {}
            if anon is not None:
                storage_options["anon"] = bool(anon)

            return zarr.storage.FSStore(
                path,
                mode=mode,
                dimension_separator="/",
                **storage_options,
            )

        return zarr.storage.NestedDirectoryStore(path)

    def open_group_at_path(self, path, mode="r", anon=None):
        store = self.make_store(path, mode=mode, anon=anon)

        if self.has_zarr3_store_api:
            if mode == "w":
                return zarr.open_group(
                    store=store,
                    mode=mode,
                    zarr_format=self.output_zarr_version,
                )

            return zarr.open_group(
                store=store,
                mode=mode,
            )

        return zarr.open_group(
            store=store,
            mode=mode,
        )

    def open_input_zarr(self, path):
        path = str(path).rstrip("/")

        if not path.startswith("s3://"):
            try:
                return self.open_group_at_path(path, mode="r", anon=None)
            except Exception:
                child = self.first_child_zarr(path, anon=None)
                if child is None:
                    raise
                return self.open_group_at_path(child, mode="r", anon=None)

        private_error = None

        try:
            return self.open_group_at_path(path, mode="r", anon=False)
        except Exception as e:
            private_error = e

        try:
            child = self.first_child_zarr(path, anon=False)
            if child is not None:
                return self.open_group_at_path(child, mode="r", anon=False)
        except Exception as e:
            private_error = e

        public_error = None

        try:
            return self.open_group_at_path(path, mode="r", anon=True)
        except Exception as e:
            public_error = e

        try:
            child = self.first_child_zarr(path, anon=True)
            if child is not None:
                return self.open_group_at_path(child, mode="r", anon=True)
        except Exception as e:
            public_error = e

        raise RuntimeError(
            "Failed to open input Zarr root or child tile zarr:\n"
            f"path={path}\n"
            f"private_error={type(private_error).__name__}: {private_error}\n"
            f"public_error={type(public_error).__name__}: {public_error}"
        )

    def copy_input_attrs(self, root):
        """
        Copy root attrs from the source zarr into the output zarr.
        """
        src_root = self.open_input_zarr(self.zarr_input_prefix)
        root.attrs.update(dict(src_root.attrs))

    def create_output_root(self):
        """
        Create output root group.
        """
        return self.open_group_at_path(
            self.output_path,
            mode="a",
            anon=False if self.output_path.startswith("s3://") else None,
        )

    def create_output_array(self, root):
        Z, Y, X = self.output_shape_zyx

        shape = (1, 1, Z, Y, X)
        chunks = (1, 1, 128, 256, 256)

        if self.has_zarr3_store_api:
            kwargs = dict(
                name="0",
                shape=shape,
                chunks=chunks,
                dtype=np.uint16,
                overwrite=False,
                fill_value=0,
            )

            if self.output_zarr_version == 2:
                root.create_array(
                    **kwargs,
                    chunk_key_encoding={
                        "name": "v2",
                        "separator": "/",
                    },
                )
                return

            root.create_array(
                **kwargs,
                chunk_key_encoding={
                    "name": "default",
                    "separator": "/",
                },
            )
            return

        # Zarr-Python 2 API.
        root.create_dataset(
            "0",
            shape=shape,
            chunks=chunks,
            dtype=np.uint16,
            overwrite=False,
            fill_value=0,
            dimension_separator="/",
        )

    def run(self):
        root = self.create_output_root()

        if "0" not in root:
            self.create_output_array(root)
