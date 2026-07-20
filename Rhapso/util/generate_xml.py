from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import boto3
import s3fs
import zarr


ZARR_PATH = "s3://aind-open-data/HCR_831990-s3-ls2_2026-05-28_00-00-00_processed_2026-05-30_03-10-11/fusion/ch_594/fused.zarr/"
OUTPUT_XML_PATH = "/Users/sean.fite/Desktop/fused_ch_594.xml"


def format_number(value) -> str:
    return f"{float(value):.12g}"


def open_zarr_group(zarr_path: str):
    root_path = zarr_path.rstrip("/")

    if root_path.startswith("s3://"):
        parsed = urlparse(root_path)
        s3_root = f"{parsed.netloc}{parsed.path}"
        s3 = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=s3_root, s3=s3, check=False)
        return zarr.open_group(store, mode="r")

    return zarr.open_group(root_path, mode="r")


def get_loader_paths(zarr_path: str) -> tuple[str | None, str, str]:
    root_path = zarr_path.rstrip("/")

    if root_path.startswith("s3://"):
        parsed = urlparse(root_path)
        object_path = Path(parsed.path.lstrip("/"))

        channel_folder = object_path.parent.name
        loader_root = object_path.parent.parent.as_posix()
        zgroup_path = f"{channel_folder}/{object_path.name}"

        return parsed.netloc, loader_root, zgroup_path

    resolved = Path(root_path).expanduser().resolve()
    loader_root = resolved.parent.parent
    zgroup_path = resolved.relative_to(loader_root).as_posix()

    return None, str(loader_root), zgroup_path


def get_channel_name(zarr_path: str) -> str:
    parsed = urlparse(zarr_path.rstrip("/"))
    channel_folder = Path(parsed.path).parent.name

    if not channel_folder.startswith("ch_"):
        raise ValueError(
            f"Could not determine channel from parent folder: {channel_folder!r}"
        )

    return channel_folder.removeprefix("ch_")


def get_spatial_unit(multiscale: dict) -> str:
    for axis in multiscale.get("axes", []):
        if (
            isinstance(axis, dict)
            and axis.get("name") in {"x", "y", "z"}
            and axis.get("unit")
        ):
            return str(axis["unit"])

    return "micrometer"


def get_level_zero_metadata(group) -> tuple[str, list[int], list[float], list[float], str]:
    multiscale = group.attrs["multiscales"][0]
    datasets = multiscale["datasets"]

    level_metadata = next(
        (
            dataset
            for dataset in datasets
            if str(dataset["path"]).strip("/") == "0"
        ),
        datasets[0],
    )

    level_path = str(level_metadata["path"]).strip("/")
    level_image = group[level_path]

    size_xyz = [int(value) for value in level_image.shape[-3:][::-1]]
    voxel_size_xyz = [1.0, 1.0, 1.0]
    origin_xyz = [0.0, 0.0, 0.0]

    for transform in level_metadata.get("coordinateTransformations", []):
        if transform.get("type") == "scale":
            voxel_size_xyz = [
                float(value)
                for value in transform["scale"][-3:][::-1]
            ]
        elif transform.get("type") == "translation":
            origin_xyz = [
                float(value)
                for value in transform["translation"][-3:][::-1]
            ]

    return (
        level_path,
        size_xyz,
        voxel_size_xyz,
        origin_xyz,
        get_spatial_unit(multiscale),
    )


def add_attribute_group(
    view_setups: ET.Element,
    name: str,
    element_name: str,
    display_name: str,
) -> None:
    group = ET.SubElement(view_setups, "Attributes", name=name)
    item = ET.SubElement(group, element_name)
    ET.SubElement(item, "id").text = "0"
    ET.SubElement(item, "name").text = display_name


def write_xml(tree: ET.ElementTree, output_xml_path: str) -> None:
    ET.indent(tree, space="\t")

    if output_xml_path.startswith("s3://"):
        parsed = urlparse(output_xml_path)
        buffer = io.BytesIO()
        tree.write(buffer, encoding="utf-8", xml_declaration=True)

        boto3.client("s3").put_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
            Body=buffer.getvalue(),
            ContentType="application/xml",
        )
        return

    output_path = Path(output_xml_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def create_fused_zarr_xml(zarr_path: str, output_xml_path: str) -> str:
    zarr_path = zarr_path.rstrip("/")
    channel_name = get_channel_name(zarr_path)
    group = open_zarr_group(zarr_path)

    level_path, size_xyz, voxel_size_xyz, origin_xyz, unit = (
        get_level_zero_metadata(group)
    )

    s3_bucket, loader_root, zgroup_path = get_loader_paths(zarr_path)
    setup_name = f"fused_ch_{channel_name}"

    root = ET.Element("SpimData", version="0.2")
    ET.SubElement(root, "BasePath", type="relative").text = "."

    sequence = ET.SubElement(root, "SequenceDescription")
    loader = ET.SubElement(
        sequence,
        "ImageLoader",
        format="bdv.multimg.zarr",
        version="1.0",
    )

    if s3_bucket:
        ET.SubElement(loader, "s3bucket").text = s3_bucket

    ET.SubElement(loader, "zarr", type="absolute").text = loader_root

    zgroups = ET.SubElement(loader, "zgroups")
    zgroup = ET.SubElement(
        zgroups,
        "zgroup",
        setup="0",
        timepoint="0",
    )
    ET.SubElement(zgroup, "path").text = zgroup_path

    view_setups = ET.SubElement(sequence, "ViewSetups")
    view_setup = ET.SubElement(view_setups, "ViewSetup")

    ET.SubElement(view_setup, "id").text = "0"
    ET.SubElement(view_setup, "name").text = setup_name
    ET.SubElement(view_setup, "size").text = " ".join(map(str, size_xyz))

    voxel_size = ET.SubElement(view_setup, "voxelSize")
    ET.SubElement(voxel_size, "unit").text = unit
    ET.SubElement(voxel_size, "size").text = " ".join(
        map(format_number, voxel_size_xyz)
    )

    attributes = ET.SubElement(view_setup, "attributes")
    ET.SubElement(attributes, "illumination").text = "0"
    ET.SubElement(attributes, "channel").text = "0"
    ET.SubElement(attributes, "tile").text = "0"
    ET.SubElement(attributes, "angle").text = "0"

    add_attribute_group(
        view_setups,
        "illumination",
        "Illumination",
        "0",
    )
    add_attribute_group(
        view_setups,
        "channel",
        "Channel",
        channel_name,
    )
    add_attribute_group(
        view_setups,
        "tile",
        "Tile",
        setup_name,
    )
    add_attribute_group(
        view_setups,
        "angle",
        "Angle",
        "0",
    )

    timepoints = ET.SubElement(
        sequence,
        "Timepoints",
        type="range",
    )
    ET.SubElement(timepoints, "first").text = "0"
    ET.SubElement(timepoints, "last").text = "0"
    ET.SubElement(sequence, "MissingViews")

    registrations = ET.SubElement(root, "ViewRegistrations")
    registration = ET.SubElement(
        registrations,
        "ViewRegistration",
        timepoint="0",
        setup="0",
    )

    voxel_x, voxel_y, voxel_z = voxel_size_xyz
    origin_x, origin_y, origin_z = origin_xyz

    transform = ET.SubElement(
        registration,
        "ViewTransform",
        type="affine",
    )
    ET.SubElement(transform, "Name").text = "calibration"
    ET.SubElement(transform, "affine").text = (
        f"{format_number(voxel_x)} 0 0 {format_number(origin_x)} "
        f"0 {format_number(voxel_y)} 0 {format_number(origin_y)} "
        f"0 0 {format_number(voxel_z)} {format_number(origin_z)}"
    )

    for section in (
        "ViewInterestPoints",
        "BoundingBoxes",
        "PointSpreadFunctions",
        "StitchingResults",
        "IntensityAdjustments",
    ):
        ET.SubElement(root, section)

    tree = ET.ElementTree(root)
    write_xml(tree, output_xml_path)

    print(f"Input Zarr: {zarr_path}")
    print(f"Loader root: {loader_root}")
    print(f"Zgroup path: {zgroup_path}")
    print(f"Level: {level_path}")
    print(f"Size XYZ: {size_xyz}")
    print(f"Voxel size XYZ: {voxel_size_xyz} {unit}")
    print(f"Origin XYZ: {origin_xyz} {unit}")
    print(f"Channel: {channel_name}")
    print(f"Output XML: {output_xml_path}")

    return output_xml_path


create_fused_zarr_xml(ZARR_PATH, OUTPUT_XML_PATH)
