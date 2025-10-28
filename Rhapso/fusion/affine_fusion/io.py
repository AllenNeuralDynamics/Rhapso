"""
Defines all standard input to fusion algorithm.
"""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import boto3
import dask.array as da
import numpy as np
from numcodecs import Blosc
import re
import s3fs
import tensorstore as ts
import xmltodict
import yaml
import zarr
import fsspec

from . import geometry


def read_config_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def write_config_yaml(yaml_path: str, yaml_data: dict) -> None:
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_data, file)


def open_zarr_s3(bucket: str, path: str) -> ts.TensorStore:
    return ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'http',
            'base_url': f'https://{bucket}.s3.us-west-2.amazonaws.com/{path}',
        },
    }).result()


class InputArray:
    def __getitem__(self, value):
        """
        Member function for slice syntax, ex: arr[0:10, 0:10]
        Value is a Python slice object.
        """
        raise NotImplementedError("Please implement in InputArray subclass.")

    @property
    def shape(self):
        raise NotImplementedError("Please implement in InputArray subclass.")


class InputDask(InputArray):
    def __init__(self, arr: da.Array):
        self.arr = arr

    def __getitem__(self, slice):
        return np.array(self.arr[slice].compute())

    @property
    def shape(self):
        return self.arr.shape


class InputTensorstore(InputArray):
    def __init__(self, arr: ts.TensorStore):
        self.arr = arr

    def __getitem__(self, slice):
        return np.array(self.arr[slice])

    @property
    def shape(self):
        return self.arr.shape


class Dataset:
    """
    Data are 5d tczyx objects.
    Transforms are 3d zyx objects.
    """

    class WriteError(Exception):
        pass

    @property
    def tile_volumes_tczyx(self) -> dict[int, InputArray]:
        """
        Dict of tile_id -> tile references.
        """
        raise NotImplementedError("Please implement in Dataset subclass.")

    @tile_volumes_tczyx.setter
    def tile_volumes_tczyx(self, value):
        raise Dataset.WriteError("tile_volumes_tczyx is read-only.")

    @property
    def tile_transforms_zyx(self) -> dict[int, list[geometry.Transform]]:
        """
        Dict of tile_id -> tile transforms.
        """
        raise NotImplementedError("Please implement in Dataset subclass.")

    @tile_transforms_zyx.setter
    def tile_transforms_zyx(self, value):
        raise Dataset.WriteError("tile_transforms_zyx is read-only.")

    @property
    def tile_resolution_zyx(self) -> tuple[float, float, float]:
        """
        Specifies absolute size of each voxel in tile volume.
        Tile resolution is used to scale tile volume into absolute space prior to registration.
        """
        raise NotImplementedError("Please implement in Dataset subclass.")

    @tile_resolution_zyx.setter
    def tile_resolution_zyx(self, value):
        raise Dataset.WriteError("tile_resolution_zyx is read-only.")


class BigStitcherDataset(Dataset):
    """
    Dataset class for loading in BigStitcher Dataset.
    Intended for the base registration channel.
    """

    def __init__(self, xml_path: str, s3_path: str, datastore: int, level: int = 0):
        self.xml_path = xml_path
        self.s3_path = s3_path

        assert datastore in [0, 1], \
            f"Only 0 = Dask and 1 = Tensorstore supported."
        self.datastore = datastore  # {0 = Dask, 1 = Tensorstore}

        allowed_levels = [0, 1, 2, 3, 4, 5]
        assert level in allowed_levels, \
            f"Level {level} is not in {allowed_levels}"
        self.level = level

        self.tile_cache: dict[int, InputArray] = {}
        self.transform_cache: dict[int, list[geometry.Transform]] = {}

    @property
    def tile_volumes_tczyx(self) -> dict[int, InputArray]:
        if len(self.tile_cache) != 0:
            tile_paths = self._extract_tile_paths(self.xml_path)
            tile_paths = self._extract_tile_paths(self.xml_path)
            for t_id, t_path in tile_paths.items():
                if not self.s3_path.endswith('/'):
                    self.s3_path = self.s3_path + '/'

                level_str = '/' + str(self.level)  # Ex: '/0'
                tile_paths[t_id] = self.s3_path + Path(t_path).name + level_str
            return self.tile_cache, tile_paths

        # Otherwise, fetch for first time
        tile_paths = self._extract_tile_paths(self.xml_path)
        for t_id, t_path in tile_paths.items():
            if not self.s3_path.endswith('/'):
                self.s3_path = self.s3_path + '/'

            level_str = '/' + str(self.level)  # Ex: '/0'
            tile_paths[t_id] = self.s3_path + Path(t_path).name + level_str

        tile_arrays: dict[int, InputArray] = {}
        for tile_id, t_path in tile_paths.items():
            
            arr = None
            if self.datastore == 0:  # Dask
                tile_zarr = da.from_zarr(t_path)
                arr = InputDask(tile_zarr)
            elif self.datastore == 1:  # Tensorstore
                # Referencing the following naming convention:
                # s3://BUCKET_NAME/DATASET_NAME/TILE/NAME/CHANNEL
                parts = t_path.split('/')
                bucket = parts[2]
                third_slash_index = len(parts[0]) + len(parts[1]) + len(parts[2]) + 3
                obj = t_path[third_slash_index:]

                tile_zarr = open_zarr_s3(bucket, obj)
                arr = InputTensorstore(tile_zarr)
            
            print(f'Loading Tile {tile_id} / {len(tile_paths)}')
            tile_arrays[int(tile_id)] = arr

        self.tile_cache = tile_arrays

        return tile_arrays, tile_paths

    @property
    def tile_transforms_zyx(self) -> dict[int, list[geometry.Transform]]:
        if len(self.transform_cache) != 0:
            return self.transform_cache

        # Otherwise, fetch for first time
        tile_tfms = self._extract_tile_transforms(self.xml_path)
        tile_net_tfms = self._calculate_net_transforms(tile_tfms)

        for tile_id, tfm in tile_net_tfms.items():
            # BigStitcher XYZ -> ZYX
            # Given Matrix 3x4:
            # Swap Rows 0 and 2; Swap Colums 0 and 2
            tmp = np.copy(tfm)
            tmp[[0, 2], :] = tmp[[2, 0], :]
            tmp[:, [0, 2]] = tmp[:, [2, 0]]
            tfm = tmp

            # Assemble matrix stack:
            # 1) Add base registration
            matrix_stack = [geometry.Affine(tfm)]

            # 2) Append up/down-sampling transforms
            sf = 2. ** self.level
            up = geometry.Affine(np.array([[sf, 0., 0., 0.],
                                           [0., sf, 0., 0.],
                                           [0., 0., sf, 0.]]))
            down = geometry.Affine(np.array([[1./sf, 0., 0., 0.],
                                             [0., 1./sf, 0., 0.],
                                             [0., 0., 1./sf, 0.]]))
            matrix_stack.insert(0, up)
            matrix_stack.append(down)
            tile_net_tfms[int(tile_id)] = matrix_stack

        self.transform_cache = tile_net_tfms

        return tile_net_tfms

    @property
    def tile_resolution_zyx(self) -> tuple[float, float, float]:
        if self.xml_path.startswith("s3://"):
            with fsspec.open(self.xml_path, mode="rt") as f:
                data: OrderedDict = xmltodict.parse(f.read())
        else:
            with open(self.xml_path, "r") as file:
                data: OrderedDict = xmltodict.parse(file.read())

        resolution_str = data["SpimData"]["SequenceDescription"]["ViewSetups"][
            "ViewSetup"
        ][0]["voxelSize"]["size"]
        resolution_xyz = [float(num) for num in resolution_str.split(" ")]
        return tuple(resolution_xyz[::-1])

    def _extract_tile_paths(self, xml_path: str) -> dict[int, str]:
        """
        Utility called in property.
        Parses BDV xml and outputs map of setup_id -> tile path.

        Parameters
        ------------------------
        xml_path: str
            Path of xml outputted from BigStitcher.

        Returns
        ------------------------
        dict[int, str]:
            Dictionary of tile ids to tile paths.
        """
        view_paths: dict[int, str] = {}

        if xml_path.startswith("s3://"):
            with fsspec.open(xml_path, mode="rt") as f:
                data: OrderedDict = xmltodict.parse(f.read())
        else:
            with open(xml_path, "r") as file:
                data: OrderedDict = xmltodict.parse(file.read())

        parent = data["SpimData"]["SequenceDescription"]["ImageLoader"][
            "zarr"
        ]["#text"]

        for i, zgroup in enumerate(
            data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"][
                "zgroup"
            ]
        ):
            view_paths[i] = parent + "/" + zgroup["@path"]

        return view_paths

    def _extract_tile_transforms(self, xml_path: str) -> dict[int, list[dict]]:
        """
        Utility called in property.
        Parses BDV xml and outputs map of setup_id -> list of transformations
        Output dictionary maps view number to list of {'@type', 'Name', 'affine'}
        where 'affine' contains the transform as string of 12 floats.

        Matrices are listed in the order of forward execution.

        Parameters
        ------------------------
        xml_path: str
            Path of xml outputted by BigStitcher.

        Returns
        ------------------------
        dict[int, list[dict]]
            Dictionary of tile ids to transform list. List entries described above.
        """

        view_transforms: dict[int, list[dict]] = {}
        
        if xml_path.startswith("s3://"):
            with fsspec.open(xml_path, mode="rt") as f:
                data: OrderedDict = xmltodict.parse(f.read())
        else:
            with open(xml_path, "r") as file:
                data: OrderedDict = xmltodict.parse(file.read())

        for view_reg in data["SpimData"]["ViewRegistrations"][
            "ViewRegistration"
        ]:
            tfm_stack = view_reg["ViewTransform"]
            if type(tfm_stack) is not list:
                tfm_stack = [tfm_stack]
            view_transforms[int(view_reg["@setup"])] = tfm_stack

        view_transforms = {
            view: tfs[::-1] for view, tfs in view_transforms.items()
        }

        return view_transforms

    def _calculate_net_transforms(
        self, view_transforms: dict[int, list[dict]]
    ) -> dict[int, geometry.Matrix]:
        """
        Utility called in property.
        Accumulate net transform and net translation for each matrix stack.
        Net translation =
            Sum of translation vectors converted into original nominal basis
        Net transform =
            Product of 3x3 matrices
        NOTE: Translational component (last column) is defined
            wrt to the DOMAIN, not codomain.
            Implementation is informed by this given.

        Parameters
        ------------------------
        view_transforms: dict[int, list[dict]]
            Dictionary of tile ids to transforms associated with each tile.

        Returns
        ------------------------
        dict[int, np.ndarray]:
            Dictionary of tile ids to net_transform.
        """

        identity_transform = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        net_transforms: dict[int, np.ndarray] = {}
        for tile_id in view_transforms:
            net_transforms[tile_id] = np.copy(identity_transform)

        for view, tfs in view_transforms.items():
            net_translation = np.zeros(3)
            net_matrix_3x3 = np.eye(3)
            curr_inverse = np.eye(3)

            for (
                tf
            ) in (
                tfs
            ):  # Tfs is a list of dicts containing transform under 'affine' key
                nums = [float(val) for val in tf["affine"].split(" ")]
                matrix_3x3 = np.array([nums[0::4], nums[1::4], nums[2::4]])
                translation = np.array(nums[3::4])

                net_translation = net_translation + (
                    curr_inverse @ translation
                )
                net_matrix_3x3 = matrix_3x3 @ net_matrix_3x3
                curr_inverse = np.linalg.inv(
                    net_matrix_3x3
                )  # Update curr_inverse

            net_transforms[view] = np.hstack(
                (net_matrix_3x3, net_translation.reshape(3, 1))
            )

        return net_transforms


class BigStitcherDatasetChannel(BigStitcherDataset):
    """
    Convenience Dataset class that reuses tile registrations,
    tile shapes, and tile resolution across channels.
    Tile volumes is overloaded with channel-specific data.

    NOTE: Only loads full resolution images/registrations.
    """

    def __init__(self, xml_path: str, s3_path: str, channel_num: int, datastore: int):
        """
        Only new information required is channel number.
        """
        super().__init__(xml_path, s3_path, datastore)
        self.channel_num = channel_num

        self.tile_cache: dict[int, InputArray] = {}

    @property
    def tile_volumes_tczyx(self) -> dict[int, InputArray]:
        """
        Load in channel-specific tiles.
        """

        if len(self.tile_cache) != 0:
            return self.tile_cache

        # Otherwise fetch for first time
        tile_arrays: dict[int, InputArray] = {}

        with open(self.xml_path, "r") as file:
            data: OrderedDict = xmltodict.parse(file.read())
        tile_id_lut = {}
        for zgroup in data['SpimData']['SequenceDescription']['ImageLoader']['zgroups']['zgroup']:
            tile_id = zgroup['@setup']
            tile_name = zgroup['path']
            s_parts = tile_name.split('_')
            location = (int(s_parts[2]),
                        int(s_parts[4]),
                        int(s_parts[6]))
            tile_id_lut[location] = int(tile_id)

        # Reference path: s3://aind-open-data/HCR_677594_2023-10-20_15-10-36/SPIM.ome.zarr/
        # Reference tilename: <tile_name, no underscores>_X_####_Y_####_Z_####_ch_###.zarr
        slash_2 = self.s3_path.find('/', self.s3_path.find('/') + 1)
        slash_3 = self.s3_path.find('/', self.s3_path.find('/', self.s3_path.find('/') + 1) + 1)
        bucket_name = self.s3_path[slash_2 + 1:slash_3]
        directory_path = self.s3_path[slash_3 + 1:]

        for p in self._list_bucket_directory(bucket_name, directory_path):
            if p.endswith('.zgroup'):
                continue

            # Data loading
            channel_num = -1
            search_result = re.search(r'(\d*)\.zarr.?$', p)
            if search_result:
                channel_num = int(search_result.group(1))
                if channel_num == self.channel_num:

                    full_resolution_p = self.s3_path + p + '/0'
                    s_parts = p.split('_')
                    location = (int(s_parts[2]),
                                int(s_parts[4]),
                                int(s_parts[6]))
                    tile_id = tile_id_lut[location]

                    arr = None
                    if self.datastore == 0:  # Dask
                        tile_zarr = da.from_zarr(full_resolution_p)
                        arr = InputDask(tile_zarr)

                    elif self.datastore == 1:  # Tensorstore
                        # Referencing the following naming convention:
                        # s3://BUCKET_NAME/DATASET_NAME/TILE/NAME/CHANNEL
                        parts = full_resolution_p.split('/')
                        bucket = parts[2]
                        third_slash_index = len(parts[0]) + len(parts[1]) + len(parts[2]) + 3
                        obj = full_resolution_p[third_slash_index:]

                        tile_zarr = open_zarr_s3(bucket, obj)
                        arr = InputTensorstore(tile_zarr)

                    print(f'Loading Tile {tile_id} / {len(tile_id_lut)}')
                    tile_arrays[int(tile_id)] = arr

        self.tile_cache = tile_arrays

        return tile_arrays

    def _list_bucket_directory(self, bucket_name: str, directory_path: str):
        client = boto3.client("s3")
        result = client.list_objects(
            Bucket=bucket_name, Prefix=directory_path, Delimiter="/"
        )

        paths = []   # These are paths
        for o in result.get("CommonPrefixes"):
            paths.append(o.get("Prefix"))

        # Parse the ending files from the paths
        files = []
        for p in paths:
            if p.endswith('/'):
                p = p.rstrip("/")  # Remove trailing slash from directories

            parts = p.split('/')
            files.append(parts[-1])

        return files


class OutputArray:
    def __setitem__(self, index, value):
        raise NotImplementedError("Please implement in InputArray subclass.")
    

class OutputDask(OutputArray): 
    def __init__(self, arr: da.Array):
        self.arr = arr

    def __setitem__(self, index, value):
        self.arr[index] = value


class OutputTensorstore(OutputArray):
    def __init__(self, arr: ts.TensorStore):
        self.arr = arr

    def __setitem__(self, index, value):
        self.arr[index].write(value).result()


@dataclass
class OutputParameters:
    path: str
    chunksize: tuple[int, int, int, int, int]
    resolution_zyx: tuple[float, float, float]
    datastore: int  # {0 == Dask, 1 == Tensorstore}
    dtype: np.dtype = np.uint16
    dimension_separator: str = "/"
    compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)


@dataclass
class RuntimeParameters:
    """
    Simplified Runtime Parameters
    option:
        0: single process exectution
        1: multiprocessing execution
        2: dask execution
    pool_size: number of processes/vCPUs for options {1, 2}
    worker_cells:
        list of cells/chunks this execution operates on
    """
    option: int
    pool_size: int
    worker_cells: list[tuple[int, int, int]]