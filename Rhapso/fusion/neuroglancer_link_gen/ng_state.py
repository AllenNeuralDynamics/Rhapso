"""
Class to represent a configuration state to visualize data in neuroglancer
"""
import re
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import xmltodict
from pint import UnitRegistry

from .ng_layer import NgLayer
from .utils import utils

# IO types
PathLike = Union[str, Path]


class NgState:
    """
    Class to represent a neuroglancer state (configuration json)
    """

    def __init__(
        self,
        input_config: dict,
        mount_service: str,
        bucket_path: str,
        output_dir: PathLike,
        verbose: Optional[bool] = False,
        base_url: Optional[str] = "https://neuroglancer-demo.appspot.com/",
        json_name: Optional[str] = "process_output.json",
        dataset_name: Optional[str] = None,
    ) -> None:
        """
        Class constructor

        Parameters
        ------------------------
        image_config: dict
            Dictionary with the json configuration based on neuroglancer docs.
        mount_service: Optional[str]
            Could be 'gs' for a bucket in Google Cloud or 's3' in Amazon.
        bucket_path: str
            Path in cloud service where the dataset will be saved
        output_dir: PathLike
            Directory where the json will be written.
        verbose: Optional[bool]
            If true, additional information will be shown. Default False.
        base_url: Optional[str]
            Neuroglancer service url
        json_name: Optional[str]
            Name of json file with neuroglancer configuration
        dataset_name: Optional[str]
            Name of the dataset. If None, the name of the output_dir directory will be used.

        """

        self.input_config = input_config
        self.output_json = Path(self.__fix_output_json_path(output_dir))
        self.verbose = verbose
        self.mount_service = mount_service
        self.bucket_path = bucket_path
        self.base_url = base_url
        self.json_name = json_name
        # Component after S3 bucket and before filename in the "ng_link" field of the output JSON
        self.dataset_name = dataset_name
        if self.dataset_name is None:
            self.dataset_name = Path(self.output_json).stem

        # State and layers attributes
        self.__state = {}
        self.__dimensions = {}
        self.__layers = []

        # Initialize principal attributes
        self.initialize_attributes(self.input_config)

    def __fix_output_json_path(self, output_json: PathLike) -> str:
        """
        Fixes the json output path to have a similar structure for all links.

        Parameters
        ------------------------
        output_json: PathLike
            Path of the json output path.

        Returns
        ------------------------
        str
            String with the fixed outputh path.
        """
        output_json = Path(
            str(output_json)
            .replace("/home/jupyter/", "")
            .replace("////", "//")
        )

        return output_json

    def __unpack_axis(
        self, axis_values: dict, dest_metric: Optional[str] = "meters"
    ) -> List:
        """
        Unpack axis voxel sizes converting them to meters.
        neuroglancer uses meters by default.

        Parameters
        ------------------------
        axis_values: dict
            Dictionary with the axis values with
            the following structure for an axis:
            e.g. for Z dimension {
                "voxel_size": 2.0,
                "unit": 'microns'
            }

        dest_metric: Optional[str]
            Destination metric to be used in neuroglancer. Default 'meters'.

        Returns
        ------------------------
        List
            List with two values, the converted quantity
            and it's metric in neuroglancer format.
        """

        if dest_metric not in ["meters", "seconds"]:
            raise NotImplementedError(
                f"{dest_metric} has not been implemented"
            )

        # Converting to desired metric
        unit_register = UnitRegistry()
        quantity = (
            axis_values["voxel_size"] * unit_register[axis_values["unit"]]
        )
        dest_quantity = quantity.to(dest_metric)

        # Neuroglancer metric
        neuroglancer_metric = None
        if dest_metric == "meters":
            neuroglancer_metric = "m"

        elif dest_metric == "seconds":
            neuroglancer_metric = "s"

        return [dest_quantity.m, neuroglancer_metric]

    @property
    def dimensions(self) -> dict:
        """
        Property getter of dimensions.

        Returns
        ------------------------
        dict
            Dictionary with neuroglancer dimensions' configuration.
        """
        return self.__dimensions

    @dimensions.setter
    def dimensions(self, new_dimensions: dict) -> None:
        """
        Set dimensions with voxel sizes for the image.

        Parameters
        ------------------------
        dimensions: dict
            Dictionary with the axis values
            with the following structure for an axis:
            e.g. for Z dimension {
                "voxel_size": 2.0,
                "unit": 'microns'
            }

        """

        if not isinstance(new_dimensions, dict):
            raise ValueError(
                f"Dimensions accepts only dict. Received: {new_dimensions}"
            )

        regex_axis = r"([x-zX-Z])$"

        for axis, axis_values in new_dimensions.items():
            if re.search(regex_axis, axis):
                self.__dimensions[axis] = self.__unpack_axis(axis_values)
            elif axis == "t":
                self.__dimensions[axis] = self.__unpack_axis(
                    axis_values, "seconds"
                )
            elif axis == "c'":
                self.__dimensions[axis] = [
                    axis_values["voxel_size"],
                    axis_values["unit"],
                ]

    @property
    def layers(self) -> List[dict]:
        """
        Property getter of layers.

        Returns
        ------------------------
        List[dict]
            List with neuroglancer layers' configuration.
        """
        return self.__layers

    @layers.setter
    def layers(self, layers: List[dict]) -> None:
        """
        Property setter of layers.

        Parameters
        ------------------------
        layers: List[dict]
            List that contains a configuration for each image layer.

        """

        if not isinstance(layers, list):
            raise ValueError(
                f"layers accepts only list. Received value: {layers}"
            )

        for layer in layers:
            config = {}

            if layer["type"] == "image":
                config = {
                    "image_config": layer,
                    "mount_service": self.mount_service,
                    "bucket_path": self.bucket_path,
                    "output_dimensions": self.dimensions,
                    "layer_type": layer["type"],
                }

            elif layer["type"] == "annotation":
                config = {
                    "annotation_source": layer["source"],
                    "annotation_locations": layer["annotations"],
                    "layer_type": layer["type"],
                    "output_dimensions": self.dimensions,
                    "limits": layer["limits"] if "limits" in layer else None,
                    "mount_service": self.mount_service,
                    "bucket_path": self.bucket_path,
                    "layer_name": layer["name"],
                }

            elif layer["type"] == "segmentation":
                config = {
                    "segmentation_source": layer["source"],
                    "tab": layer["tab"],
                    "layer_name": layer["name"],
                    "mount_service": self.mount_service,
                    "bucket_path": self.bucket_path,
                    "layer_type": layer["type"],
                }

            self.__layers.append(NgLayer().create(config).layer_state)

    @property
    def state(self, new_state: dict) -> None:
        """
        Property setter of state.

        Parameters
        ------------------------
        input_config: dict
            Dictionary with the configuration for the neuroglancer state

        """
        self.__state = dict(new_state)

    @state.getter
    def state(self) -> dict:
        """
        Property getter of state.

        Returns
        ------------------------
        dict
            Dictionary with the actual layer state.
        """

        actual_state = {}
        actual_state["ng_link"] = self.get_url_link()
        actual_state["dimensions"] = {}

        # Getting actual state for all attributes
        for axis, value_list in self.__dimensions.items():
            actual_state["dimensions"][axis] = value_list

        actual_state["layers"] = self.__layers

        actual_state["showAxisLines"] = True
        actual_state["showScaleBar"] = True

        return actual_state

    def initialize_attributes(self, input_config: dict) -> None:
        """
        Initializes the following attributes for a given
        image layer: dimensions, layers.

        Parameters
        ------------------------
        input_config: dict
            Dictionary with the configuration for each image layer

        """

        # Initializing dimension
        self.dimensions = input_config["dimensions"]

        # Initializing layers
        self.layers = input_config["layers"]

        # Initializing state
        self.__state = self.state

        for key, val in input_config.items():
            if key == "showAxisLines":
                self.show_axis_lines = val

            elif key == "showScaleBar":
                self.show_scale_bar = val

            elif key == "title":
                self.title = val

            elif key == "crossSectionOrientation":
                self.cross_section_orientation = val

            elif key == "crossSectionScale":
                self.cross_section_scale = val

            elif key == "projectionScale":
                self.projection_scale = val

            elif key == "layout":
                self.layout = val

            elif key == "position":
                self.position = val

    @property
    def title(self) -> str:
        """
        Getter of the title property

        Returns
        ------------------------
        str
            String value of the title.
        """
        return self.__state["title"]

    @title.setter
    def title(self, new_title: str) -> None:
        """
        Sets the title parameter in neuroglancer link.

        Parameters
        ------------------------
        new_title: str
            String that will appear in the browser tab title.

        Raises
        ------------------------
        ValueError:
            If the parameter is not an string.
        """
        self.__state["title"] = str(new_title)

    @property
    def cross_section_scale(self) -> float:
        """
        Getter of the cross_section_scale property

        Returns
        ------------------------
        float
            Value of the cross_section_scale.
        """
        return self.__state["crossSectionScale"]

    @cross_section_scale.setter
    def cross_section_scale(self, new_cross_section_scale: float) -> None:
        """
        Sets the cross_section_scale parameter in neuroglancer link.

        Parameters
        ------------------------
        new_cross_section_scale: float
            Cross section scale value for the neuroglancer state.

        Raises
        ------------------------
        ValueError:
            If the parameter is not an float.
        """
        self.__state["crossSectionScale"] = float(new_cross_section_scale)

    @property
    def projection_scale(self) -> float:
        """
        Getter of the projection_scale property

        Returns
        ------------------------
        float
            Value of the projection_scale.
        """
        return self.__state["projectionScale"]

    @projection_scale.setter
    def projection_scale(self, new_scale: float) -> None:
        """
        Sets the projection_scale parameter in neuroglancer link.

        Parameters
        ------------------------
        new_scale: float
            Projection scale value for the neuroglancer state.

        Raises
        ------------------------
        ValueError:
            If the parameter is not an float.
        """
        self.__state["projectionScale"] = float(new_scale)

    @property
    def cross_section_orientation(self) -> List[float]:
        """
        Getter of the cross_section_orientation property

        Returns
        ------------------------
        List[float]
            List of values to set the cross section orientation
        """
        return self.__state["crossSectionOrientation"]

    @cross_section_orientation.setter
    def cross_section_orientation(self, new_orientation: List[float]) -> None:
        """
        Sets the cross_section_orientation parameter in neuroglancer link.

        Parameters
        ------------------------
        new_orientation: List[float]
            Cross section orientation values for the neuroglancer state.

        Raises
        ------------------------
        ValueError:
            If the list contents are not float.
        """
        new_orientation = [float(i) for i in new_orientation]
        self.__state["crossSectionOrientation"] = new_orientation

    @property
    def layout(self) -> str:
        """
        Getter of the layout property.
        This specifies panel layout in neuroglancer, such as '4panel',
        'xz', 'zx', etc.

        Returns
        ------------------------
        str
            Viewer panel layout.
        """
        return self.__state["layout"]

    @layout.setter
    def layout(self, new_layout: str) -> None:
        """
        Sets the layout parameter in neuroglancer link.

        Parameters
        ------------------------
        new_layout: str
            Neuroglancer viewer panels layout.
            Must be one of:
             - 4panel
             - 3d
             - xy, yx, xz, etc.

        Raises
        ------------------------
        ValueError:
            If the string is not one of the defined choices
        """
        available_layouts = [
            k[0] + k[1] for k in combinations(["x", "y", "z"], 2)
        ]
        available_layouts += [i[::-1] for i in available_layouts]
        available_layouts += ["3d", "4panel"]

        if new_layout not in available_layouts:
            raise ValueError(f"Viewer layout {new_layout} is not valid")
        else:
            self.__state["layout"] = new_layout

    @property
    def position(self) -> List[float]:
        """
        Getter of the position property

        Returns
        ------------------------
        List[float]
            List of values of the position
        """
        return self.__state["position"]

    @position.setter
    def position(self, new_position: List[float]):
        """
        Sets the viewer's center position.

        Parameters
        ------------------------
        new_position: List[float]
            List of coordinates to center on.
            If the list is shorter than the number of axes, the viewer 
            will be positioned at the center of the unset axes.

        Raises
        ------------------------
        ValueError:
            If the list contents are not float.
        """
        new_position = [float(i) for i in new_position]
        self.__state["position"] = new_position

    @property
    def show_axis_lines(self) -> bool:
        """
        Getter of the show axis lines property

        Returns
        ------------------------
        bool
            Boolean with the show axis lines value.
        """
        return self.__state["showAxisLines"]

    @show_axis_lines.setter
    def show_axis_lines(self, new_show_axis_lines: bool) -> None:
        """
        Sets the visible parameter in neuroglancer link.

        Parameters
        ------------------------
        new_show_axis_lines: bool
            Boolean that dictates if the image axis are visible or not.

        Raises
        ------------------------
        ValueError:
            If the parameter is not an boolean.
        """
        self.__state["showAxisLines"] = bool(new_show_axis_lines)

    @property
    def show_scale_bar(self) -> bool:
        """
        Getter of the show scale bar property

        Returns
        ------------------------
        bool
            Boolean with the show scale bar value.
        """
        return self.__state["showScaleBar"]

    @show_scale_bar.setter
    def show_scale_bar(self, new_show_scale_bar: bool) -> None:
        """
        Sets the visible parameter in neuroglancer link.

        Parameters
        ------------------------
        new_show_scale_bar: bool
            Boolean that dictates if the image scale bar are visible or not.

        Raises
        ------------------------
        ValueError:
            If the parameter is not an boolean.
        """
        self.__state["showScaleBar"] = bool(new_show_scale_bar)

    def save_state_as_json(self, update_state: Optional[bool] = False) -> None:
        """
        Saves a neuroglancer state as json.

        Parameters
        ------------------------
        update_state: Optional[bool]
            Updates the neuroglancer state with dimensions
            and layers in case they were changed using
            class methods. Default False
        """

        if update_state:
            self.__state = self.state

        final_path = Path(self.output_json).joinpath(self.json_name)
        utils.save_dict_as_json(final_path, self.__state, verbose=self.verbose)

    def get_url_link(self) -> str:
        """
        Creates the neuroglancer link based on where the json will be written.

        Returns
        ------------------------
        str
            Neuroglancer url to visualize data.
        """

        json_path = f"{self.mount_service}://{self.bucket_path}/{self.dataset_name}/{self.json_name}"

        link = f"{self.base_url}#!{json_path}"

        return link


def get_points_from_xml(path: PathLike, encoding: str = "utf-8") -> List[dict]:
    """
    Function to parse the points from the
    cell segmentation capsule.

    Parameters
    -----------------

    Path: PathLike
        Path where the XML is stored.

    encoding: str
        XML encoding. Default: "utf-8"

    Returns
    -----------------
    List[dict]
        List with the location of the points.
    """

    with open(path, "r", encoding=encoding) as xml_reader:
        xml_file = xml_reader.read()

    xml_dict = xmltodict.parse(xml_file)
    cell_data = xml_dict["CellCounter_Marker_File"]["Marker_Data"][
        "Marker_Type"
    ]["Marker"]

    new_cell_data = []
    for cell in cell_data:
        new_cell_data.append(
            {"x": cell["MarkerX"], "y": cell["MarkerY"], "z": cell["MarkerZ"],}
        )

    return new_cell_data


def smartspim_example():
    """
    Example one related to the SmartSPIM data
    """
    example_data = {
        "dimensions": {
            # check the order
            "z": {"voxel_size": 2.0, "unit": "microns"},
            "y": {"voxel_size": 1.8, "unit": "microns"},
            "x": {"voxel_size": 1.8, "unit": "microns"},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "position": [1900.5, 4400.5, 3800.5, 0.5],
        "crossSectionOrientation": [0.5, 0.5, -0.5, 0.5],
        "crossSectionScale": 10.0,
        "projectionOrientation": [0.641, 0.660, 0.004, 0.391],
        "projectionScale": 13000.0,
        "layers": [
            {
                "source": "image_path.zarr",
                "type": "image",
                "channel": 0,
                # 'name': 'image_name_0',
                "shader": {"color": "green", "emitter": "RGB", "vec": "vec3"},
                "shaderControls": {
                    "normalized": {"range": [0, 500]}
                },  # Optional
            },
            {
                "source": "image_path.zarr",
                "type": "image",
                "channel": 1,
                # 'name': 'image_name_1',
                "shader": {"color": "red", "emitter": "RGB", "vec": "vec3"},
                "shaderControls": {
                    "normalized": {"range": [0, 500]}
                },  # Optional
            },
        ],
    }

    neuroglancer_link = NgState(
        input_config=example_data,
        mount_service="s3",
        bucket_path="aind-msma-data",
        output_dir="/Users/camilo.laiton/repositories/aind-ng-link/src",
    )

    data = neuroglancer_link.state
    print(data)
    # neuroglancer_link.save_state_as_json('test.json')
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())


def exaspim_example():
    """
    Example 2 related to the ExaSPIM data
    """
    example_data = {
        "dimensions": {
            # check the order
            "x": {"voxel_size": 0.74800002019210531934, "unit": "microns"},
            "y": {"voxel_size": 0.74800002019210531934, "unit": "microns"},
            "z": {"voxel_size": 1, "unit": "microns"},
            "c'": {"voxel_size": 1, "unit": ""},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": [
            {
                "type": "image",  # Optional
                "source": [
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0000_y_0000_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -14192,
                            "delta_y": -10640,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0000_y_0001_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -14192,
                            "delta_y": -19684.000456947142,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0000_y_0002_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -14192,
                            "delta_y": -28727.998694435275,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0001_y_0000_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -26255.200652782467,
                            "delta_y": -10640,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0001_y_0001_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -26255.200652782467,
                            "delta_y": -19684.000456947142,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0001_y_0002_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -26255.200652782467,
                            "delta_y": -28727.998694435275,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0002_y_0000_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -38318.39686664473,
                            "delta_y": -10640,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0002_y_0001_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -38318.39686664473,
                            "delta_y": -19684.000456947142,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0002_y_0002_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -38318.39686664473,
                            "delta_y": -28727.998694435275,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0003_y_0000_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -50381.5952999671,
                            "delta_y": -10640,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0003_y_0001_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -50381.5952999671,
                            "delta_y": -19684.000456947142,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0003_y_0002_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -50381.5952999671,
                            "delta_y": -28727.998694435275,
                            "delta_z": 0,
                        },
                    },
                ],
                "channel": 0,  # Optional
                "shaderControls": {
                    "normalized": {"range": [30, 70]}
                },  # Optional
                "visible": True,  # Optional
                "opacity": 0.50,
            }
        ],
        "showScaleBar": False,
        "showAxisLines": False,
    }

    neuroglancer_link = NgState(
        input_config=example_data,
        mount_service="s3",
        bucket_path="aind-msma-data",
        output_dir="/Users/camilo.laiton/repositories/aind-ng-link/src",
    )

    data = neuroglancer_link.state
    # print(data)
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())


def example_3(cells):
    """
    Example 3 with the annotation layer
    """
    example_data = {
        "dimensions": {
            # check the order
            "z": {"voxel_size": 2.0, "unit": "microns"},
            "y": {"voxel_size": 1.8, "unit": "microns"},
            "x": {"voxel_size": 1.8, "unit": "microns"},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": [
            {
                "source": "image_path.zarr",
                "type": "image",
                "channel": 0,
                # 'name': 'image_name_0',
                "shader": {"color": "green", "emitter": "RGB", "vec": "vec3"},
                "shaderControls": {
                    "normalized": {"range": [0, 500]}
                },  # Optional
            },
            {
                "type": "annotation",
                "source": "precomputed:///Users/camilo.laiton/repositories/aind-ng-link/src/precomputed",
                "tool": "annotatePoint",
                "name": "annotation_name_layer",
                "annotations": cells,
                # Pass None or delete limits if
                # you want to include all the points
                # "limits": [100, 200],  # None # erase line
            },
        ],
    }

    neuroglancer_link = NgState(
        input_config=example_data,
        mount_service="s3",
        bucket_path="aind-msma-data",
        output_dir="/Users/camilo.laiton/repositories/aind-ng-link/src",
    )

    data = neuroglancer_link.state
    print(data)
    # neuroglancer_link.save_state_as_json('test.json')
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())


def dispim_example():
    """
    Example related to the dispim data
    """

    def generate_source_list(
        s3_path: str,
        channel_name: str,
        camera_index: str,
        n_tiles: int,
        affine_transform: list,
        translation_deltas: list,
    ) -> list:
        """
        Example to generate layers with
        an affine transformation

        Parameters
        ----------
        s3_path: str
            Path in S3 where the images are stored

        channel_name: str
            Channel name of the dataset

        camera_index: str
            Camera index of the dataset

        n_tiles: int
            Number of tiles in the dataset

        affine_transform: list
            List with the affine transformation
            that will be applied in the data

        translation_deltas: list
            List with the translation per axis
            xyz

        Returns
        -------
        list
            List with the source layers for
            neuroglancer
        """
        multisource_layer = []
        n_rows = 5
        # Affine transformation without translation
        # and in ng format tczyx usually, check output dims.

        list_n_tiles = range(0, n_tiles + 1)

        shift = 1

        if camera_index:
            list_n_tiles = range(n_tiles, -1, -1)
            shift = -1

        new_affine_transform = affine_transform.copy()

        for n_tile in list_n_tiles:
            n_tile = str(n_tile)

            if len(n_tile) == 1:
                n_tile = "0" + str(n_tile)

            tile_name = f"{s3_path}/tile_X_00{n_tile}_Y_0000_Z_0000_CH_{channel_name}_cam{camera_index}.zarr"

            if n_tile:
                start_point = n_rows - 1

                new_translation_deltas = list(
                    map(
                        lambda delta: delta * shift * int(n_tile),
                        translation_deltas,
                    )
                )

                # Setting translations for axis
                for delta in new_translation_deltas:
                    new_affine_transform[start_point][-1] = delta
                    start_point -= 1

            else:
                new_affine_transform = affine_transform.copy()

            multisource_layer.append(
                {
                    "url": tile_name,
                    "transform_matrix": new_affine_transform.tolist(),
                }
            )

        return multisource_layer

    # t  c  z  y  x  T
    ng_affine_transform = np.zeros((5, 6), np.float16)
    np.fill_diagonal(ng_affine_transform, 1)

    theta = 45

    # Adding shearing
    shearing_zx = np.tan(np.deg2rad(theta))
    ng_affine_transform[2, 4] = shearing_zx

    translation_x = 0
    translation_y = 1140
    translation_z = 0

    # Parameters
    s3_path = (
        "s3://aind-open-data/diSPIM_647459_2022-12-07_00-00-00/diSPIM.zarr"
    )
    channel_names = ["0405", "0488", "0561"]
    colors = ["#3f2efe", "#58fea1", "#f15211"]
    camera_indexes = [0]  # , 1]
    n_tiles = 13  # 13
    layers = []
    visible = True

    for camera_index in camera_indexes:
        if camera_index == 1:
            # Mirror Z stack and apply same angle for cam0
            ng_affine_transform[2, 2] = -1

        # elif camera_index ==  1:
        #     # No mirror for camera 1
        #     ng_affine_transform[2, 2] = 1

        for channel_name_idx in range(len(channel_names)):
            layers.append(
                {
                    "type": "image",  # Optional
                    "source": generate_source_list(
                        s3_path=s3_path,
                        channel_name=channel_names[channel_name_idx],
                        camera_index=camera_index,
                        n_tiles=n_tiles,
                        affine_transform=ng_affine_transform,
                        translation_deltas=[
                            translation_x,
                            translation_y,
                            translation_z,
                        ],
                    ),
                    "channel": 0,  # Optional
                    "shaderControls": {
                        "normalized": {"range": [0, 800]}
                    },  # Optional
                    "shader": {
                        "color": colors[channel_name_idx],
                        "emitter": "RGB",
                        "vec": "vec3",
                    },
                    "visible": visible,  # Optional
                    "opacity": 0.50,
                    "name": f"CH_{channel_names[channel_name_idx]}_CAM{camera_index}",
                }
            )

    example_data = {
        "dimensions": {
            # check the order
            "x": {"voxel_size": 0.298, "unit": "microns"},
            "y": {"voxel_size": 0.298, "unit": "microns"},
            "z": {"voxel_size": 0.176, "unit": "microns"},
            "c'": {"voxel_size": 1, "unit": ""},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": layers,
        "showScaleBar": False,
        "showAxisLines": False,
    }

    neuroglancer_link = NgState(
        input_config=example_data,
        mount_service="s3",
        bucket_path="aind-msma-data",
        output_dir="/Users/camilo.laiton/repositories/aind-ng-link/src",
    )

    data = neuroglancer_link.state
    # print(data)
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())


# flake8: noqa: E501
def examples():
    """
    Examples of how to use the neurglancer state class.
    """
    # example_1()

    # Transformation matrix can be a dictionary with the axis translations
    # or a affine transformation (list of lists)
    # example_2()

    cells_path = "/Users/camilo.laiton/Downloads/detected_cells (5).xml"
    cells = get_points_from_xml(cells_path)
    example_3(cells)

    # dispim_example()


if __name__ == "__main__":
    examples()
