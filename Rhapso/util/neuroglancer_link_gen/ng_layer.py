"""
Class to represent a layer of a configuration state to visualize images in neuroglancer
"""
import inspect
import json
import multiprocessing
import os
import struct
import time
from multiprocessing.managers import BaseManager, NamespaceProxy
from pathlib import Path
from typing import Dict, List, Optional, Union, get_args

import neuroglancer
import numpy as np

from .utils import shader_utils, utils

# IO types
PathLike = Union[str, Path]
SourceLike = Union[PathLike, List[Dict]]


class ObjProxy(NamespaceProxy):
    """Returns a proxy instance for any user defined data-type. The proxy instance will have the namespace and
    functions of the data-type (except private/protected callables/attributes). Furthermore, the proxy will be
    pickable and can its state can be shared among different processes."""

    @classmethod
    def populate_obj_attributes(cls, real_cls):
        """
        Populates attributes of the proxy object
        """
        DISALLOWED = set(dir(cls))
        ALLOWED = [
            "__sizeof__",
            "__eq__",
            "__ne__",
            "__le__",
            "__repr__",
            "__dict__",
            "__lt__",
            "__gt__",
        ]
        DISALLOWED.add("__class__")
        new_dict = {}
        for attr, value in inspect.getmembers(real_cls, callable):
            if attr not in DISALLOWED or attr in ALLOWED:
                new_dict[attr] = cls._proxy_wrap(attr)
        return new_dict

    @staticmethod
    def _proxy_wrap(attr):
        """
        This method creates function that calls the proxified object's method.
        """

        def f(self, *args, **kwargs):
            """
            Function that calls the proxified object's method.
            """
            return self._callmethod(attr, args, kwargs)

        return f


def buf_builder(x, y, z, buf_):
    """builds the buffer"""
    pt_buf = struct.pack("<3f", x, y, z)
    buf_.extend(pt_buf)


attributes = ObjProxy.populate_obj_attributes(bytearray)
bytearrayProxy = type("bytearrayProxy", (ObjProxy,), attributes)


def generate_precomputed_cells(cells, path, res):
    """
    Function for saving precomputed annotation layer

    Parameters
    -----------------

    cells: dict
        output of the xmltodict function for importing cell locations
    path: str
        path to where you want to save the precomputed files
    res: neuroglancer.CoordinateSpace()
        data on the space that the data will be viewed
    buf: bytearrayProxy object
        if you want to use multiprocessing set to bytearrayProxy object else
        leave as None

    """

    BaseManager.register(
        "bytearray",
        bytearray,
        bytearrayProxy,
        exposed=tuple(dir(bytearrayProxy)),
    )
    manager = BaseManager()
    manager.start()

    buf = manager.bytearray()

    cell_list = []
    for cell in cells:
        cell_list.append([int(cell["z"]), int(cell["y"]), int(cell["x"])])

    l_bounds = np.min(cell_list, axis=0)
    u_bounds = np.max(cell_list, axis=0)

    output_path = os.path.join(path, "spatial0")
    utils.create_folder(output_path)

    metadata = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": res.to_json(),
        "lower_bound": [float(x) for x in l_bounds],
        "upper_bound": [float(x) for x in u_bounds],
        "annotation_type": "point",
        "properties": [],
        "relationships": [],
        "by_id": {"key": "by_id",},
        "spatial": [
            {
                "key": "spatial0",
                "grid_shape": [1] * res.rank,
                "chunk_size": [max(1, float(x)) for x in u_bounds - l_bounds],
                "limit": len(cell_list),
            },
        ],
    }

    with open(os.path.join(path, "info"), "w") as f:
        f.write(json.dumps(metadata))

    with open(os.path.join(output_path, "0_0_0"), "wb") as outfile:
        start_t = time.time()

        total_count = len(cell_list)  # coordinates is a list of tuples (x,y,z)

        print("Running multiprocessing")

        if not isinstance(buf, type(None)):
            buf.extend(struct.pack("<Q", total_count))

            with multiprocessing.Pool(processes=os.cpu_count()) as p:
                p.starmap(
                    buf_builder, [(x, y, z, buf) for (x, y, z) in cell_list]
                )

            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack(
                "<%sQ" % len(cell_list), *range(len(cell_list))
            )
            buf.extend(id_buf)
        else:
            buf = struct.pack("<Q", total_count)

            for x, y, z in cell_list:
                pt_buf = struct.pack("<3f", x, y, z)
                buf += pt_buf

            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack(
                "<%sQ" % len(cell_list), *range(len(cell_list))
            )
            buf += id_buf

        print(
            "Building file took {0} minutes".format(
                (time.time() - start_t) / 60
            )
        )

        outfile.write(bytes(buf))


def helper_create_ng_translation_matrix(
    delta_x: Optional[float] = 0,
    delta_y: Optional[float] = 0,
    delta_z: Optional[float] = 0,
    n_cols: Optional[int] = 6,
    n_rows: Optional[int] = 5,
) -> List:
    """
    Helper function to create the translation matrix based on deltas over each axis

    Parameters
    ------------------------
    delta_x: Optional[float]
        Translation over the x axis.
    delta_y: Optional[float]
        Translation over the y axis.
    delta_z: Optional[float]
        Translation over the z axis.
    n_cols: Optional[int]
        number of columns to create the translation matrix.
    n_rows: Optional[int]
        number of rows to create the translation matrix.

    Raises
    ------------------------
    ValueError:
        Raises if the N size of the transformation matrix is not
        enough for the deltas.

    Returns
    ------------------------
    List
        List with the translation matrix
    """

    translation_matrix = np.zeros((n_rows, n_cols), np.float16)
    np.fill_diagonal(translation_matrix, 1)

    deltas = [delta_x, delta_y, delta_z]
    start_point = n_rows - 1

    if start_point < len(deltas):
        raise ValueError(
            "N size of transformation matrix is not enough for deltas"
        )

    # Setting translations for axis
    for delta in deltas:
        translation_matrix[start_point][-1] = delta
        start_point -= 1

    return translation_matrix.tolist()


def helper_reverse_dictionary(dictionary: dict) -> dict:
    """
    Helper to reverse a dictionary

    Parameters
    ------------------------
    dictionary: dict
        Dictionary to reverse

    Returns
    ------------------------
    dict
        Reversed dictionary
    """

    keys = list(dictionary.keys())
    values = list(dictionary.values())
    new_dict = {}

    for idx in range(len(keys) - 1, -1, -1):
        new_dict[keys[idx]] = values[idx]

    return new_dict


class SegmentationLayer:
    """
    Class to represent a neuroglancer segmentation layer in the
    configuration json
    """

    def __init__(
        self,
        segmentation_source: PathLike,
        tab: str,
        layer_name: str,
        mount_service: str,
        bucket_path: str,
        layer_type: Optional[str] = "segmentation",
    ) -> None:
        """
        Class constructor

        Parameters
        ------------------------
        segmentation_source: PathLike
            Segmentation layer path

        tab: str
            Tab name

        layer_name: str
            Layer name

        mount_service: Optional[str]
            This parameter could be 'gs' referring to a bucket in Google Cloud or 's3'in Amazon.

        bucket_path: str
            Path in cloud service where the dataset will be saved

        mount_service: Optional[str]
            This parameter could be 'gs' referring to a bucket in Google Cloud or 's3'in Amazon.

        bucket_path: str
            Path in cloud service where the dataset will be saved

        layer_type: str
            Layer type. Default: segmentation
        """

        self.__layer_state = {}
        self.segmentation_source = segmentation_source
        self.tab_name = tab
        self.layer_name = layer_name
        self.mount_service = mount_service
        self.bucket_path = bucket_path
        self.layer_type = layer_type

        # Optional parameter that must be used when we have multiple images per layer
        # Dictionary needs to be reversed for correct visualization
        self.update_state()

    def __set_s3_path(self, orig_source_path: PathLike) -> str:
        """
        Private method to set a s3 path based on a source path.
        Available image formats: ['.zarr']

        Parameters
        ------------------------
        orig_source_path: PathLike
            Source path of the image

        Raises
        ------------------------
        NotImplementedError:
            Raises if the image format is not zarr.

        Returns
        ------------------------
        str
            String with the source path pointing to the mount service in the cloud
        """

        s3_path = None
        if not orig_source_path.startswith(f"{self.mount_service}://"):
            orig_source_path = Path(orig_source_path)
            s3_path = (
                f"{self.mount_service}://{self.bucket_path}/{orig_source_path}"
            )

        else:
            s3_path = orig_source_path

        return s3_path

    def set_segmentation_source(self, source: PathLike) -> dict:
        """
        Sets the segmentation source.

        Parameters
        ---------------
        source: PathLike
            Path where the precomputed format is
            located

        Returns
        ---------------
        dict:
            Dictionary with the modified layer.
        """

        actual_state = self.__layer_state
        
        if isinstance(source, str) and 'precomputed://' in source:
            write_path = Path(source.replace("precomputed://", ""))
            s3_path = self.__set_s3_path(str(write_path))

            actual_state["source"] = f"precomputed://{s3_path}"
            
            return actual_state

        elif isinstance(source, dict):
            layer_dict = {}
            for k, v in source.items():
                if k == "url":
                    write_path = Path(v.replace("precomputed://", ""))
                    s3_path = self.__set_s3_path(str(write_path))
                    
                    layer_dict[k] = s3_path
                elif k == "transform":
                    layer_dict[k] = v
 
            actual_state["source"] = layer_dict

        else:
            raise NotImplementedError("This option has not been implemented")
            
        return actual_state

    def set_tool(self, tool_name: str) -> dict:
        """
        Sets the tool name in neuroglancer.

        Parameters
        ---------------
        tool_name: str
            Tool name in neuroglancer.

        Returns
        ---------------
        dict:
            Dictionary with the modified layer.
        """
        actual_state = self.__layer_state
        actual_state["tool"] = str(tool_name)
        return actual_state

    def set_tab_name(self, tab_name: str) -> dict:
        """
        Sets the tab name in neuroglancer.

        Parameters
        ---------------
        tab_name: str
            Tab name in neuroglancer.

        Returns
        ---------------
        dict:
            Dictionary with the modified layer.
        """
        actual_state = self.__layer_state
        actual_state["tab"] = str(tab_name)
        return actual_state

    def set_layer_name(self, layer_name: str) -> dict:
        """
        Sets the layer name

        Parameters
        ---------------
        layer_name: str
            Layer name

        Returns
        ---------------
        dict:
            Dictionary with the modified layer.
        """
        actual_state = self.__layer_state
        actual_state["name"] = layer_name
        return actual_state

    def update_state(self):
        """
        Updates the state of the layer
        """

        self.__layer_state = self.set_segmentation_source(
            self.segmentation_source
        )

        self.__layer_state = self.set_tab_name(self.tab_name)

        self.__layer_state = self.set_layer_name(self.layer_name)

        self.__layer_state["type"] = "segmentation"

    @property
    def layer_state(self) -> dict:
        """
        Getter of layer state property.

        Returns
        ------------------------
        dict:
            Dictionary with the current configuration of the layer state.
        """
        return self.__layer_state

    @layer_state.setter
    def layer_state(self, new_layer_state: dict) -> None:
        """
        Setter of layer state property.

        Parameters
        ------------------------
        new_layer_state: dict
            Dictionary with the new configuration of the layer state.
        """
        self.__layer_state = dict(new_layer_state)


class AnnotationLayer:
    """
    Class to represent a neuroglancer annotation layer in the
    configuration json
    """

    def __init__(
        self,
        annotation_source: Union[str, dict],
        annotation_locations: List[dict],
        output_dimensions: dict,
        mount_service: str,
        bucket_path: str,
        layer_type: Optional[str] = "annotation",
        limits: Optional[List[int]] = None,
        layer_name: Optional[str] = "annotationLayer",
    ) -> None:
        """
        Class constructor

        Parameters
        ------------------------
        annotation_source: Union[str, dict]
            Location of the annotation layer information

        annotation_locations: List[dict]
            List with the location of the points. The dictionary
            must have this order: {"x": valx, "y": valy, "z": valz}

        output_dimensions: dict
            Dictionary with the output dimensions of the layer.
            Note: The axis order indicates where the points
            will be placed.

        mount_service: Optional[str]
            This parameter could be 'gs' referring to a bucket in Google Cloud or 's3'in Amazon.

        bucket_path: str
            Path in cloud service where the dataset will be saved

        mount_service: Optional[str]
            This parameter could be 'gs' referring to a bucket in Google Cloud or 's3'in Amazon.

        bucket_path: str
            Path in cloud service where the dataset will be saved

        layer_type: str
            Layer type. Default: annotation

        limits: Optional[List[int]]
            Range of points to visualize

        layer_name: Optional[str]
            Layer name
        """

        self.__layer_state = {}
        self.annotation_source = annotation_source
        self.annotation_locations = annotation_locations
        self.mount_service = mount_service
        self.bucket_path = bucket_path
        self.layer_type = layer_type
        self.limits = limits
        self.layer_name = layer_name

        # Optional parameter that must be used when we have multiple images per layer
        # Dictionary needs to be reversed for correct visualization
        self.output_dimensions = (
            output_dimensions  # helper_reverse_dictionary(output_dimensions)
        )
        self.update_state()

    def __set_s3_path(self, orig_source_path: PathLike) -> str:
        """
        Private method to set a s3 path based on a source path.
        Available image formats: ['.zarr']

        Parameters
        ------------------------
        orig_source_path: PathLike
            Source path of the image

        Raises
        ------------------------
        NotImplementedError:
            Raises if the image format is not zarr.

        Returns
        ------------------------
        str
            String with the source path pointing to the mount service in the cloud
        """

        s3_path = None
        if not orig_source_path.startswith(f"{self.mount_service}://"):
            orig_source_path = Path(orig_source_path)
            s3_path = (
                f"{self.mount_service}://{self.bucket_path}/{orig_source_path}"
            )

        else:
            s3_path = orig_source_path

        return s3_path

    def set_annotation_source(
        self, source: Union[str, dict], output_dimensions: dict
    ) -> dict:
        """
        Sets the annotation source.

        Parameters
        ---------------
        source: Union[str, dict]
            Dictionary with the annotation source

        Returns
        ---------------
        dict:
            Dictionary with the modified layer.
        """

        actual_state = self.__layer_state

        if "precomputed://" in source:
            axis = list(output_dimensions.keys())
            values = list(output_dimensions.values())

            names = []
            units = []
            scales = []

            for axis_idx in range(len(axis)):
                if axis[axis_idx] == "t" or axis[axis_idx] == "c'":
                    continue

                names.append(axis[axis_idx])
                scales.append(values[axis_idx][0])
                units.append(values[axis_idx][1])

            write_path = Path(source.replace("precomputed://", ""))

            coord_space = neuroglancer.CoordinateSpace(
                names=names, units=units, scales=scales
            )

            print("Write path: ", write_path)

            # Generates the precomputed format
            generate_precomputed_cells(
                self.annotation_locations, write_path, coord_space
            )

            s3_path = self.__set_s3_path(str(write_path))

            actual_state["source"] = f"precomputed://{s3_path}"

        else:
            actual_state["source"] = source
            actual_state = self.__set_transform(self.output_dimensions)

        return actual_state

    def __set_transform(
        self, layer_state: dict, output_dimensions: dict
    ) -> dict:
        """
        Sets the output dimensions and transformation
        to the annotation layer.

        Parameters
        ---------------
        output_dimensions: dict
            Dictionary with the output dimensions
            for the layer. The order of the axis in
            the dictionary determines the location
            of the points. {"t": t, "c": c, "z", z, ...}

        Returns
        ---------------
        dict:
            Dictionary with the modified layer.
        """

        actual_state = layer_state.copy()
        actual_state["source"]["transform"] = {
            "outputDimensions": output_dimensions
        }

        return actual_state

    def set_tool(self, tool_name: str) -> dict:
        """
        Sets the tool name in neuroglancer.

        Parameters
        ---------------
        tool_name: str
            Tool name in neuroglancer.

        Returns
        ---------------
        dict:
            Dictionary with the modified layer.
        """
        actual_state = self.__layer_state
        actual_state["tool"] = str(tool_name)
        return actual_state

    def set_tab_name(self, tab_name: str) -> dict:
        """
        Sets the tab name in neuroglancer.

        Parameters
        ---------------
        tab_name: str
            Tab name in neuroglancer.

        Returns
        ---------------
        dict:
            Dictionary with the modified layer.
        """
        actual_state = self.__layer_state
        actual_state["tab"] = str(tab_name)
        return actual_state

    def set_annotations(
        self,
        annotation_points: List[Dict[str, int]],
        annotation_type: str,
        limits: Optional[List[int]] = None,
    ) -> dict:
        """
        Sets the annotations in neuroglancer using a
        list with the locations of the annotations.

        Parameters
        ---------------
        annotation_points: List[Dict[str, int]]
            Points where the annotations will
            be placed.

        annotation_type: str
            Annotation type. e.g., "points"

        limits: Optional[List[int]]
            Limist of points. [lower_limit, upper_limit]

        Returns
        ---------------
        dict:
            Dictionary with the modified layer.
        """

        # Conditional to add specific points in
        # visualization link

        annotation_len = len(annotation_points)
        lower_limit = 0
        upper_limit = 0

        if limits is None:
            upper_limit = annotation_len
            lower_limit = 0

        else:
            upper_limit = limits[1]
            lower_limit = limits[0]

        if not isinstance(upper_limit, int):
            upper_limit = annotation_len

        if not isinstance(lower_limit, int) or lower_limit < 0:
            lower_limit = 0

        if (
            upper_limit <= 0
            or upper_limit < lower_limit
            or upper_limit > annotation_len
        ):
            raise ValueError("Limits must be in a valid range.")

        actual_state = self.__layer_state

        if annotation_type == "points":

            def get_point_config(id: str, point: Dict[str, int]) -> dict:
                """
                Gets the point configuration for neuroglancer

                Parameters
                --------------
                id: str
                    Unique ID to represent a point

                point: Dict[str, int]
                    Point location

                Returns
                ---------------
                dict:
                    Dictionary with the point configuration
                    adapted to neuroglancer.
                """

                dimension_order = self.output_dimensions.keys()

                point_list = []
                tc_missing = len(dimension_order) - 3

                if tc_missing < 0:
                    raise ValueError("Expected number of dimensions: 3")

                # Decrease # of iterations by setting it by default
                for axis in dimension_order:
                    if axis in point:
                        point_list.append(float(point[axis]))

                    else:
                        point_list.append(float(0.5))

                point_config = {
                    "point": point_list,
                    "type": "point",
                    "id": str(id),
                }

                return point_config

            actual_state["annotations"] = []

            for annotation_point_idx in range(lower_limit, upper_limit):
                point_config = get_point_config(
                    annotation_point_idx,
                    annotation_points[annotation_point_idx],
                )

                actual_state["annotations"].append(point_config)

        return actual_state

    def set_layer_name(self, layer_name: str) -> dict:
        """
        Sets the layer name

        Parameters
        ---------------
        layer_name: str
            Layer name

        Returns
        ---------------
        dict:
            Dictionary with the modified layer.
        """
        actual_state = self.__layer_state
        actual_state["name"] = layer_name
        return actual_state

    def update_state(self):
        """
        Updates the state of the layer
        """

        self.__layer_state = self.set_annotation_source(
            self.annotation_source, self.output_dimensions
        )

        if isinstance(self.annotation_source, dict):
            self.__layer_state = self.set_annotations(
                self.annotation_locations, "points", self.limits
            )

        self.__layer_state = self.set_tool("annotatePoint")

        self.__layer_state = self.set_tab_name("annotations")

        self.__layer_state = self.set_layer_name(self.layer_name)

        self.__layer_state["type"] = "annotation"

    @property
    def layer_state(self) -> dict:
        """
        Getter of layer state property.

        Returns
        ------------------------
        dict:
            Dictionary with the current configuration of the layer state.
        """
        return self.__layer_state

    @layer_state.setter
    def layer_state(self, new_layer_state: dict) -> None:
        """
        Setter of layer state property.

        Parameters
        ------------------------
        new_layer_state: dict
            Dictionary with the new configuration of the layer state.
        """
        self.__layer_state = dict(new_layer_state)


class ImageLayer:
    """
    Class to represent a neuroglancer image layer in the
    configuration json
    """

    def __init__(
        self,
        image_config: dict,
        mount_service: str,
        bucket_path: str,
        layer_type: Optional[str] = "image",
        output_dimensions: Optional[dict] = None,
    ) -> None:
        """
        Class constructor

        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration based on neuroglancer documentation.
        mount_service: Optional[str]
            This parameter could be 'gs' referring to a bucket in Google Cloud or 's3'in Amazon.
        bucket_path: str
            Path in cloud service where the dataset will be saved
        layer_type: Optional[str]
            Image type based on neuroglancer documentation.

        """

        self.__layer_state = {}
        self.image_config = image_config
        self.mount_service = mount_service
        self.bucket_path = bucket_path
        self.layer_type = layer_type

        # Optional parameter that must be used when we have multiple images per layer
        # Dictionary needs to be reversed for correct visualization
        self.output_dimensions = helper_reverse_dictionary(output_dimensions)

        # Fix image source
        self.image_source = self.__fix_image_source(image_config["source"])
        image_config["source"] = self.image_source

        self.update_state(image_config)

    def __set_s3_path(self, orig_source_path: PathLike) -> str:
        """
        Private method to set a s3 path based on a source path.
        Available image formats: ['.zarr']

        Parameters
        ------------------------
        orig_source_path: PathLike
            Source path of the image

        Raises
        ------------------------
        NotImplementedError:
            Raises if the image format is not zarr.

        Returns
        ------------------------
        str
            String with the source path pointing to the mount service in the cloud
        """

        s3_path = None
        if not orig_source_path.startswith(f"{self.mount_service}://"):
            # Work with code ocean
            if "/scratch/" in orig_source_path:
                orig_source_path = orig_source_path.replace("/scratch/", "")

            elif "/results/" in orig_source_path:
                orig_source_path = orig_source_path.replace("/results/", "")

            orig_source_path = Path(orig_source_path)
            s3_path = (
                f"{self.mount_service}://{self.bucket_path}/{orig_source_path}"
            )

        else:
            s3_path = orig_source_path

        if s3_path.endswith(".zarr"):
            s3_path = "zarr://" + s3_path

        else:
            raise NotImplementedError(
                "This format has not been implemented yet for visualization"
            )

        return s3_path

    def __set_sources_paths(self, sources_paths: List) -> List:
        """
        Private method to set multiple image sources on s3 path. It also accepts
        a transformation matrix that should be provided in the form of a list for
        or a affine transformation or dictionary for a translation matrix.
        Available image formats: ['.zarr']

        Parameters
        ------------------------
        sources_paths: List
            List of dictionaries with the image sources and its transformation
            matrices in the case they are provided.

        Returns
        ------------------------
        List
            List of dictionaries with the configuration for neuroglancer
        """
        new_source_path = []

        for source in sources_paths:
            new_dict = {}

            for key in source.keys():
                if key == "transform_matrix" and isinstance(
                    source["transform_matrix"], dict
                ):
                    new_dict["transform"] = {
                        "matrix": helper_create_ng_translation_matrix(
                            delta_x=source["transform_matrix"]["delta_x"],
                            delta_y=source["transform_matrix"]["delta_y"],
                            delta_z=source["transform_matrix"]["delta_z"],
                        ),
                        "outputDimensions": self.output_dimensions,
                    }

                elif key == "transform_matrix" and isinstance(
                    source["transform_matrix"], list
                ):
                    new_dict["transform"] = {
                        "matrix": source["transform_matrix"],
                        "outputDimensions": self.output_dimensions,
                    }

                elif key == "url":
                    new_dict["url"] = self.__set_s3_path(source["url"])

                else:
                    new_dict[key] = source[key]

            new_source_path.append(new_dict)

        return new_source_path

    def __fix_image_source(self, source_path: SourceLike) -> str:
        """
        Fixes the image source path to include the type of image neuroglancer accepts.

        Parameters
        ------------------------
        source_path: SourceLike
            Path or list of paths where the images are located with their transformation matrix.

        Returns
        ------------------------
        SourceLike
            Fixed path(s) for neuroglancer json configuration.
        """
        new_source_path = None

        if isinstance(source_path, list):
            # multiple sources in single image
            new_source_path = self.__set_sources_paths(source_path)

        elif isinstance(source_path, get_args(PathLike)):
            # Single source image
            new_source_path = self.__set_s3_path(source_path)

        return new_source_path

    # flake8: noqa: C901
    def set_default_values(
        self, image_config: dict = {}, overwrite: bool = False
    ) -> None:
        """
        Set default values for the image.

        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration. Similar to self.image_config

        overwrite: bool
            If the parameters already have values, with this flag they can be overwritten.

        """

        if overwrite:
            self.image_channel = 0
            self.shader_control = {"normalized": {"range": [0, 200]}}
            self.visible = True
            self.__layer_state["name"] = str(Path(self.image_source).stem)
            self.__layer_state["type"] = str(self.layer_type)

        elif len(image_config):
            # Setting default image_config in json image layer
            if "channel" not in image_config:
                # Setting channel to 0 for image
                self.image_channel = 0

            if "shaderControls" not in image_config:
                self.shader_control = {"normalized": {"range": [0, 200]}}

            if "visible" not in image_config:
                self.visible = True

            if "name" not in image_config:
                try:
                    channel = self.__layer_state["localDimensions"]["c'"][0]

                except KeyError:
                    channel = ""

                if isinstance(self.image_source, get_args(PathLike)):
                    self.__layer_state[
                        "name"
                    ] = f"{Path(self.image_source).stem}_{channel}"

                else:
                    self.__layer_state[
                        "name"
                    ] = f"{Path(self.image_source[0]['url']).stem}_{channel}"

            if "type" not in image_config:
                self.__layer_state["type"] = str(self.layer_type)

    # flake8: noqa: C901
    def update_state(self, image_config: dict) -> None:
        """
        Set default values for the image.

        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration. Similar to self.image_config
            e.g.: image_config = {
                'type': 'image', # Optional
                'source': 'image_path',
                'channel': 0, # Optional
                'name': 'image_name', # Optional
                'shader': {
                    'color': 'green',
                    'emitter': 'RGB',
                    'vec': 'vec3'
                },
                'shaderControls': { # Optional
                    "normalized": {
                        "range": [0, 200]
                    }
                }
            }
        """

        for param, value in image_config.items():
            if param in ["type", "name", "blend"]:
                self.__layer_state[param] = str(value)

            if param in ["visible"]:
                self.visible = value

            if param == "shader":
                self.shader = self.__create_shader(value)

            if param == "channel":
                self.image_channel = value

            if param == "shaderControls":
                self.shader_control = value

            if param == "opacity":
                self.opacity = value

            if param == "source":
                if isinstance(value, get_args(PathLike)):
                    self.__layer_state[param] = str(value)

                elif isinstance(value, list):
                    # Setting list of dictionaries with image configuration
                    self.__layer_state[param] = value

        self.set_default_values(image_config)

    def __create_shader(self, shader_config: dict) -> str:
        """
        Creates a configuration for the neuroglancer shader.

        Parameters
        ------------------------
        shader_config: dict
            Configuration of neuroglancer's shader.

        Returns
        ------------------------
        str
            String with the shader configuration for neuroglancer.
        """

        monochrome_keys = set(["color", "emitter", "vec"])
        rgb_keys = set(["r_range", "g_range", "b_range"])
        config_keys = set(shader_config.keys())
        if config_keys == monochrome_keys:
            return shader_utils.create_monochrome_shader(
                color=shader_config["color"],
                emitter=shader_config["emitter"],
                vec=shader_config["vec"],
            )
        elif config_keys == rgb_keys:
            return shader_utils.create_rgb_shader(
                r_range=shader_config["r_range"],
                g_range=shader_config["g_range"],
                b_range=shader_config["b_range"],
            )
        else:
            raise RuntimeError(
                f"Do not know how to create shader code for shader_config "
                f"with keys {list(shader_config.keys())}"
            )

    @property
    def opacity(self) -> str:
        """
        Getter of the opacity property

        Returns
        ------------------------
        str
            String with the opacity value
        """
        return self.__layer_state["opacity"]

    @opacity.setter
    def opacity(self, opacity: float) -> None:
        """
        Sets the opacity parameter in neuroglancer link.

        Parameters
        ------------------------
        opacity: float
            Float number between [0-1] that indicates the opacity.

        Raises
        ------------------------
        ValueError:
            If the parameter is not an boolean.
        """
        self.__layer_state["opacity"] = float(opacity)

    @property
    def shader(self) -> str:
        """
        Getter of the shader property

        Returns
        ------------------------
        str
            String with the shader value
        """
        return self.__layer_state["shader"]

    @shader.setter
    def shader(self, shader_config: str) -> None:
        """
        Sets a configuration for the neuroglancer shader.

        Parameters
        ------------------------
        shader_config: str
            Shader configuration for neuroglancer in string format.
            e.g. #uicontrol vec3 color color(default=\"green\")\n#uicontrol invlerp normalized\nvoid main() {\n  emitRGB(color * normalized());\n}

        Raises
        ------------------------
        ValueError:
            If the provided shader_config is not a string.

        """
        self.__layer_state["shader"] = str(shader_config)

    @property
    def shader_control(self) -> dict:
        """
        Getter of the shader control property

        Returns
        ------------------------
        str
            String with the shader control value
        """
        return self.__layer_state["shaderControls"]

    @shader_control.setter
    def shader_control(self, shader_control_config: dict) -> None:
        """
        Sets a configuration for the neuroglancer shader control.

        Parameters
        ------------------------
        shader_control_config: dict
            Shader control configuration for neuroglancer.

        Raises
        ------------------------
        ValueError:
            If the provided shader_control_config is not a dictionary.

        """
        self.__layer_state["shaderControls"] = dict(shader_control_config)

    @property
    def image_channel(self) -> int:
        """
        Getter of the current image channel in the layer

        Returns
        ------------------------
        int
            Integer with the current image channel
        """
        return self.__layer_state["localDimensions"]["c"]

    @image_channel.setter
    def image_channel(self, channel: int) -> None:
        """
        Sets the image channel in case the file contains multiple channels.

        Parameters
        ------------------------
        channel: int
            Channel position. It will be incremented in 1 since neuroglancer channels starts in 1.

        Raises
        ------------------------
        ValueError:
            If the provided channel is not an integer.

        """
        self.__layer_state["localDimensions"] = {"c'": [int(channel) + 1, ""]}

    @property
    def visible(self) -> bool:
        """
        Getter of the visible attribute of the layer.
        True means the layer will be visible when the image
        is loaded in neuroglancer, False otherwise.

        Returns
        ------------------------
        bool
            Boolean with the current visible value
        """
        return self.__layer_state["visible"]

    @visible.setter
    def visible(self, visible: bool) -> None:
        """
        Sets the visible parameter in neuroglancer link.

        Parameters
        ------------------------
        visible: bool
            Boolean that dictates if the image is visible or not.

        Raises
        ------------------------
        ValueError:
            If the parameter is not an boolean.
        """
        self.__layer_state["visible"] = bool(visible)

    @property
    def layer_state(self) -> dict:
        """
        Getter of layer state property.

        Returns
        ------------------------
        dict:
            Dictionary with the current configuration of the layer state.
        """
        return self.__layer_state

    @layer_state.setter
    def layer_state(self, new_layer_state: dict) -> None:
        """
        Setter of layer state property.

        Parameters
        ------------------------
        new_layer_state: dict
            Dictionary with the new configuration of the layer state.
        """
        self.__layer_state = dict(new_layer_state)


class NgLayer:
    """
    Class to represent a neuroglancer layer in the configuration json
    """

    def __init__(self) -> None:
        """
        Class constructor
        """
        self.__extensions = ["image", "annotation", "segmentation"]

        self.factory = {
            "image": ImageLayer,
            "annotation": AnnotationLayer,
            "segmentation": SegmentationLayer,
        }

    @property
    def extensions(self) -> List:
        """
        Method to return the allowed format extensions of the layers.
        Returns
        ------------------------
        List
            List with the allowed layers format extensions
        """
        return self.__extensions

    def create(self, params: dict):
        """
        Instantiates the class corresponding to
        the type of annotation.
        """

        layer_type = params["layer_type"]

        if layer_type not in self.__extensions:
            raise NotImplementedError(
                f"Layer type {layer_type} has not been implemented"
            )

        return self.factory[layer_type](**params)
