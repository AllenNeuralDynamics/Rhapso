from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from urllib.parse import urlparse
from ome_zarr.format import CurrentFormat, FormatV04
from ome_zarr.writer import write_multiscales_metadata
from zarr.errors import ContainsGroupError
import numpy as np
import zarr
import math
import fsspec

"""
Handles OME-NGFF + OMERO metadata, axes, transforms and Zarr group helpers.
"""

class OMEMetadata:
    def __init__(self, zarr_path: str, dataset_shape: List[int], scale_factor, voxel_size: List[float], n_lvls: int, 
                 chunk_size: List[int], channel_names: Optional[List[str]] = None, origin: Optional[List[float]] = None) -> None:
        self.zarr_path = zarr_path
        self.dataset_shape = dataset_shape
        self.scale_factors = self.normalize_scale_factors(scale_factor)
        self.voxel_size = [float(v) for v in voxel_size]
        self.n_lvls = n_lvls
        self.chunk_size = chunk_size
        self.channel_names = channel_names
        self.origin = origin or [0.0, 0.0, 0.0]
        self.zarr_format = None

    @staticmethod
    def is_s3_path(path: str) -> bool:
        parsed = urlparse(str(path))
        return parsed.scheme == "s3"

    @staticmethod
    def get_parent_path(path: str) -> str:
        parsed = urlparse(path)
        if parsed.scheme == "s3":
            parts = parsed.path.strip("/").split("/")
            parent_key = "/".join(parts[:-1])
            return f"s3://{parsed.netloc}/{parent_key}"
        return str(Path(path).parent)

    @staticmethod
    def safe_create_zarr_group(store, path: str = "", **kwargs):
        try:
            return zarr.group(store=store, path=path, overwrite=False, **kwargs)
        except ContainsGroupError:
            return zarr.open_group(store=store, path=path, mode="r+")
    
    @staticmethod
    def path_has_zarr_metadata(path: str, metadata_name: str) -> bool:
        path = str(path).rstrip("/")
        fs, fs_path = fsspec.core.url_to_fs(path)
        return fs.exists(f"{fs_path.rstrip('/')}/{metadata_name}")

    @classmethod
    def detect_existing_zarr_format(cls, path: str) -> int:
        """
        Detect zarr format from an existing zarr group/array path.
        """
        path = str(path).rstrip("/")

        if cls.path_has_zarr_metadata(path, "zarr.json"):
            return 3

        if (
            cls.path_has_zarr_metadata(path, ".zgroup")
            or cls.path_has_zarr_metadata(path, ".zarray")
        ):
            return 2

        raise RuntimeError(
            "Could not detect existing zarr format. Expected an existing "
            f"v2 or v3 zarr at: {path}"
        )
    
    def open_root_and_channel_group(self, output_path: str):
        output_path = str(output_path).rstrip("/")
        stack_name = Path(output_path).name

        zarr_format = self.detect_existing_zarr_format(output_path)
        self.zarr_format = zarr_format

        if self.is_s3_path(output_path):
            store = zarr.storage.FsspecStore.from_url(
                output_path,
                storage_options={},
                read_only=False,
            )
        else:
            store = zarr.storage.LocalStore(
                output_path,
                read_only=False,
            )

        output_group = zarr.open_group(
            store=store,
            mode="r+",
            zarr_format=zarr_format,
        )

        print(
            f"[OMEMetadata] Opened output zarr root: {output_path} "
            f"(zarr v{zarr_format})"
        )

        return output_group, stack_name

    @staticmethod
    def normalize_scale_factors(scale_factor):
        """
        Ensure scale_factor is a list-of-lists[int, int, int] or list[int].
        """
        if isinstance(scale_factor[0], (list, tuple)):
            return [[int(v) for v in lvl] for lvl in scale_factor]
        return [int(s) for s in scale_factor]

    @staticmethod
    def make_channel_metadata(dataset_shape: List[int]):
        """
        Default min/max + start/end + colors for all channels.
        """
        n_channels = dataset_shape[1]
        channel_minmax = [(0.0, 1.0) for _ in range(n_channels)]
        channel_startend = [(0.0, 1.0) for _ in range(n_channels)]
        channel_colors = [i for i in range(n_channels)]
        return channel_minmax, channel_startend, channel_colors

    @staticmethod
    def get_pyramid_metadata() -> Dict[str, Any]:
        return {
            "metadata": {
                "description": "Downscaling using the windowed mean",
                "method": "xarray_multiscale.reducers.windowed_mean",
                "version": "unknown",
                "args": "[false]",
                "kwargs": {},
            }
        }

    @staticmethod
    def custom_compute_scales(scale_factors_zyx: List[Tuple[int, int, int]], pixelsizes: Tuple[float, float, float], 
                              chunks: Tuple[int, int, int, int, int], data_shape: Tuple[int, int, int, int, int], 
                              translation: Optional[List[float]] = None) -> Tuple[List, List]:
        transforms = [
            [
                {
                    "type": "scale",
                    "scale": [
                        1.0,
                        1.0,
                        pixelsizes[0],
                        pixelsizes[1],
                        pixelsizes[2],
                    ],
                }
            ]
        ]
        if translation is not None:
            transforms[0].append(
                {"type": "translation", "translation": translation}
            )

        chunk_sizes = []
        lastz = data_shape[2]
        lasty = data_shape[3]
        lastx = data_shape[4]
        opts = dict(
            chunks=(
                1,
                1,
                min(lastz, chunks[2]),
                min(lasty, chunks[3]),
                min(lastx, chunks[4]),
            )
        )
        chunk_sizes.append(opts)

        for scale_factor in scale_factors_zyx:
            last_transform = transforms[-1][0]
            last_scale = cast(List, last_transform["scale"])
            transforms.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            1.0,
                            1.0,
                            last_scale[2] * scale_factor[0],
                            last_scale[3] * scale_factor[1],
                            last_scale[4] * scale_factor[2],
                        ],
                    }
                ]
            )
            if translation is not None:
                transforms[-1].append(
                    {"type": "translation", "translation": translation}
                )
            lastz = int(math.ceil(lastz / scale_factor[0]))
            lasty = int(math.ceil(lasty / scale_factor[1]))
            lastx = int(math.ceil(lastx / scale_factor[2]))
            opts = dict(
                chunks=(
                    1,
                    1,
                    min(lastz, chunks[2]),
                    min(lasty, chunks[3]),
                    min(lastx, chunks[4]),
                )
            )
            chunk_sizes.append(opts)

        return transforms, chunk_sizes

    @staticmethod
    def _downscale_origin(array_shape: List[int], origin: List[float], voxel_size: List[float], scale_factors, n_levels: int):
        current_shape = np.array(array_shape[-3:], dtype=np.int32)
        current_origin = np.array(origin[-3:], dtype=np.float64)
        current_voxel_size = np.array(voxel_size[-3:], dtype=np.float64)

        if isinstance(scale_factors[0], (list, tuple, np.ndarray)):
            factors = list(scale_factors)
            if len(factors) < (n_levels - 1):
                raise ValueError(
                    f"scale_factors has {len(factors)} levels, "
                    f"but n_levels={n_levels} requires at least {n_levels - 1}."
                )
            per_level = [[int(x) for x in v[-3:]] for v in factors[: n_levels - 1]]
            per_level_sf = np.array(per_level, dtype=np.int32)
        else:
            base = np.array([int(x) for x in scale_factors[-3:]], dtype=np.int32)
            per_level_sf = np.tile(base, (n_levels - 1, 1))

        new_origins = [[0.0, 0.0] + current_origin.tolist()]

        for level in range(n_levels - 1):
            sf = per_level_sf[level]
            center_shift = (current_voxel_size * (sf - 1)) / 2.0
            current_origin += center_shift
            current_shape = np.ceil(current_shape / sf).astype(int)
            current_voxel_size = current_voxel_size * sf
            new_origins.append([0.0, 0.0] + current_origin.tolist())

        return new_origins

    @staticmethod
    def _get_axes_5d(time_unit: str = "millisecond", space_unit: str = "micrometer") -> List[Dict]:
        return [
            {"name": "t", "type": "time", "unit": time_unit},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": space_unit},
            {"name": "y", "type": "space", "unit": space_unit},
            {"name": "x", "type": "space", "unit": space_unit},
        ]

    @staticmethod
    def _build_ome(data_shape: Tuple[int, ...], image_name: str, channel_names: Optional[List[str]] = None, 
                   channel_colors: Optional[List[int]] = None, channel_minmax: Optional[List[Tuple[float, float]]] = None, 
                   channel_startend: Optional[List[Tuple[float, float]]] = None) -> Dict:
        if channel_names is None:
            channel_names = [
                f"Channel:{image_name}:{i}" for i in range(data_shape[1])
            ]
        if channel_colors is None:
            channel_colors = [i for i in range(data_shape[1])]
        if channel_minmax is None:
            channel_minmax = [(0.0, 1.0) for _ in range(data_shape[1])]
        if channel_startend is None:
            channel_startend = channel_minmax

        ch = []
        for i in range(data_shape[1]):
            ch.append(
                {
                    "active": True,
                    "coefficient": 1,
                    "color": f"{channel_colors[i]:06x}",
                    "family": "linear",
                    "inverted": False,
                    "label": channel_names[i],
                    "window": {
                        "end": float(channel_startend[i][1]),
                        "max": float(channel_minmax[i][1]),
                        "min": float(channel_minmax[i][0]),
                        "start": float(channel_startend[i][0]),
                    },
                }
            )

        return {
            "id": 1,
            "name": image_name,
            "version": "0.4",
            "channels": ch,
            "rdefs": {
                "defaultT": 0,
                "defaultZ": data_shape[2] // 2,
                "model": "color",
            },
        }

    def write_ome_ngff_metadata(
        self,
        group: zarr.Group,
        arr_shape: List[int],
        final_chunksize: List[int],
        image_name: str,
        n_lvls: int,
        scale_factors,
        voxel_size,
        channel_names: Optional[List[str]] = None,
        channel_colors: Optional[List[int]] = None,
        channel_minmax: Optional[List[Tuple[float, float]]] = None,
        channel_startend: Optional[List[Tuple[float, float]]] = None,
        origin: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if metadata is None:
            metadata = {}
        # fmt = CurrentFormat()
        if self.zarr_format == 2:
            fmt = FormatV04()
        elif self.zarr_format == 3:
            fmt = CurrentFormat()
        else:
            raise RuntimeError(
                f"Unknown zarr format for OME metadata writing: {self.zarr_format}"
            )

        ome_json = self._build_ome(
            tuple(arr_shape),
            image_name,
            channel_names=channel_names,
            channel_colors=channel_colors,
            channel_minmax=channel_minmax,
            channel_startend=channel_startend,
        )
        group.attrs["omero"] = ome_json
        axes_5d = self._get_axes_5d()

        if origin is not None:
            origin_scaled = self._downscale_origin(
                arr_shape,
                origin[-3:],
                voxel_size[-3:],
                scale_factors,
                n_lvls,
            )
        else:
            origin_scaled = None

        coordinate_transformations, _ = self.custom_compute_scales(
            scale_factors, voxel_size, tuple(final_chunksize), tuple(arr_shape), None
        )

        if origin_scaled is not None:
            for level_idx, transforms in enumerate(coordinate_transformations):
                if level_idx < len(origin_scaled):
                    transforms[:] = [
                        t for t in transforms if t.get("type") != "translation"
                    ]
                    transforms.append(
                        {
                            "type": "translation",
                            "translation": origin_scaled[level_idx],
                        }
                    )

        fmt.validate_coordinate_transformations(
            len(arr_shape), n_lvls, coordinate_transformations
        )

        datasets = [{"path": str(i)} for i in range(n_lvls)]
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

        write_multiscales_metadata(group, datasets, fmt, axes_5d, **metadata)

    def run(self):
        """
        Use instance config to open groups and write OME-NGFF metadata.
        Returns: (channel_group, stack_name, scale_factors)
        """
        channel_group, stack_name = self.open_root_and_channel_group(self.zarr_path)
        channel_minmax, channel_startend, channel_colors = (self.make_channel_metadata(self.dataset_shape))

        self.write_ome_ngff_metadata(
            group=channel_group,
            arr_shape=self.dataset_shape,
            final_chunksize=self.chunk_size,
            image_name=stack_name,
            n_lvls=self.n_lvls,
            scale_factors=self.scale_factors,
            voxel_size=self.voxel_size,
            channel_names=self.channel_names,
            channel_colors=channel_colors,
            channel_minmax=channel_minmax,
            channel_startend=channel_startend,
            origin=self.origin,
            metadata=self.get_pyramid_metadata(),
        )

        print("[OMEMetadata] Wrote OME-NGFF metadata")
        return channel_group, stack_name, self.scale_factors, self.zarr_format