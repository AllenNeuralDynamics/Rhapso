import zarr
import fsspec
import s3fs
import os
import numpy as np
import copy
import xml.etree.ElementTree as ET

class XMLPrep():
    def __init__(self, fixed_image_multiscale_root, moving_image_multiscale_root, moving_segmentation_zarr_path, 
                 minimum_alignment_blocksize_physical_units, min_alignment_level):
        self.fixed_image_multiscale_root = fixed_image_multiscale_root
        self.moving_image_multiscale_root = moving_image_multiscale_root
        self.moving_segmentation_zarr_path = moving_segmentation_zarr_path
        self.minimum_alignment_blocksize_physical_units = minimum_alignment_blocksize_physical_units
        self.min_alignment_level = min_alignment_level

    def set_xml_calibration(self, xml_path, calibration_xyz):
        if xml_path.startswith("s3://"):
            with fsspec.open(xml_path, "rb") as file:
                tree = ET.parse(file)
        else:
            tree = ET.parse(xml_path)

        root = tree.getroot()
        x, y, z = calibration_xyz

        for registration in root.findall("./ViewRegistrations/ViewRegistration"):
            transform = next(
                (
                    t for t in registration.findall("ViewTransform")
                    if t.findtext("Name") == "calibration"
                ),
                None,
            )

            if transform is None:
                transform = ET.Element("ViewTransform", type="affine")
                ET.SubElement(transform, "Name").text = "calibration"
                registration.insert(0, transform)

            affine = transform.find("affine")
            if affine is None:
                affine = ET.SubElement(transform, "affine")

            affine.text = f"{x} 0 0 0 0 {y} 0 0 0 0 {z} 0"

        ET.indent(tree, space="  ")

        if xml_path.startswith("s3://"):
            with fsspec.open(xml_path, "wb") as file:
                tree.write(file, encoding="UTF-8", xml_declaration=True)
        else:
            tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    
    def get_level0_spacing_zyx(self, image_root):
        root = zarr.open(fsspec.get_mapper(image_root.rstrip("/")), mode="r")
        level_meta = next(
            d for d in root.attrs["multiscales"][0]["datasets"]
            if str(d["path"]).strip("/") == "0"
        )
        scale = next(
            t for t in level_meta["coordinateTransformations"]
            if t["type"] == "scale"
        )
        return tuple(float(v) for v in scale["scale"][-3:])

    def normalize_loop0_calibration(self, fixed_detected_xml_path, moving_detected_xml_path):
        fixed_spacing = self.get_level0_spacing_zyx(self.fixed_image_multiscale_root)
        moving_spacing = self.get_level0_spacing_zyx(self.moving_image_multiscale_root)

        scale_zyx = tuple(f / m for f, m in zip(fixed_spacing, moving_spacing))
        moving_z_scale = moving_spacing[0] / moving_spacing[1]

        fixed_cal_xyz = (
            scale_zyx[2],
            scale_zyx[1],
            scale_zyx[0] * moving_z_scale,
        )
        moving_cal_xyz = (1.0, 1.0, moving_z_scale)

        self.set_xml_calibration(fixed_detected_xml_path, fixed_cal_xyz)
        self.set_xml_calibration(moving_detected_xml_path, moving_cal_xyz)

        print(f"Fixed calibration XYZ: {fixed_cal_xyz}")
        print(f"Moving calibration XYZ: {moving_cal_xyz}")
    
    def get_split_ids(self, xml_path):
        if xml_path.startswith("s3://"):
            with fsspec.open(xml_path, "rb") as file:
                root = ET.parse(file).getroot()
        else:
            root = ET.parse(xml_path).getroot()

        return [
            int(definition.findtext("NewId"))
            for definition in root.findall(
                "./SequenceDescription/ImageLoader/SetupIds/SetupIdDefinition"
            )
            if int(definition.findtext("OldId")) == 0
        ]
  
    def renumber_fixed_setup(self, fixed_xml_path, new_id):
        if fixed_xml_path.startswith("s3://"):
            with fsspec.open(fixed_xml_path, "rb") as file:
                tree = ET.parse(file)
        else:
            tree = ET.parse(fixed_xml_path)

        root = tree.getroot()
        new_id = str(new_id)

        for zgroup in root.findall("./SequenceDescription/ImageLoader/zgroups/zgroup"):
            zgroup.set("setup", new_id)

        root.find("./SequenceDescription/ViewSetups/ViewSetup/id").text = new_id

        for registration in root.findall("./ViewRegistrations/ViewRegistration"):
            registration.set("setup", new_id)

        ET.indent(tree, space="  ")

        if fixed_xml_path.startswith("s3://"):
            with fsspec.open(fixed_xml_path, "wb") as file:
                tree.write(file, encoding="UTF-8", xml_declaration=True)
        else:
            tree.write(fixed_xml_path, encoding="UTF-8", xml_declaration=True)

        print(f"Renumbered fixed setup ID to {new_id}")

    def remove_empty_moving_tiles(
        self,
        moving_split_xml_path,
        scale_level,
        exclusion_percentage,
    ):
        if moving_split_xml_path.startswith("s3://"):
            with fsspec.open(moving_split_xml_path, "rb") as file:
                tree = ET.parse(file)
        else:
            tree = ET.parse(moving_split_xml_path)

        root = tree.getroot()
        tile_bboxes_zyx = {}

        for definition in root.findall(
            "./SequenceDescription/ImageLoader/SetupIds/SetupIdDefinition"
        ):
            if int(definition.findtext("OldId")) != 0:
                continue

            tile_id = int(definition.findtext("NewId"))
            min_xyz = tuple(map(int, definition.findtext("min").split()))
            max_xyz = tuple(map(int, definition.findtext("max").split()))

            tile_bboxes_zyx[tile_id] = (
                tuple(reversed(min_xyz)),
                tuple(value + 1 for value in reversed(max_xyz)),
            )

        seg_root = zarr.open(
            fsspec.get_mapper(
                self.moving_segmentation_zarr_path.rstrip("/")
            ),
            mode="r",
        )

        if hasattr(seg_root, "shape"):
            seg = seg_root
        else:
            level = str(scale_level)

            if level not in seg_root:
                raise KeyError(
                    f"Segmentation level {level} is missing. "
                    f"Available levels: {list(seg_root.array_keys())}"
                )

            seg = seg_root[level]

        moving_root = zarr.open(
            fsspec.get_mapper(
                self.moving_image_multiscale_root.rstrip("/")
            ),
            mode="r",
        )

        moving = (
            moving_root
            if hasattr(moving_root, "shape")
            else moving_root["0"]
        )

        moving_shape_zyx = np.asarray(
            moving.shape[-3:],
            dtype=float,
        )
        seg_shape_zyx = np.asarray(
            seg.shape[-3:],
            dtype=int,
        )

        downsample_zyx = moving_shape_zyx / seg_shape_zyx
        empty_tile_ids = []

        for tile_id, (
            tile_min_zyx,
            tile_max_zyx,
        ) in tile_bboxes_zyx.items():
            seg_min = np.floor(
                np.asarray(tile_min_zyx) / downsample_zyx
            ).astype(int)

            seg_max = np.ceil(
                np.asarray(tile_max_zyx) / downsample_zyx
            ).astype(int)

            seg_min = np.maximum(seg_min, 0)
            seg_max = np.minimum(seg_max, seg_shape_zyx)

            margin = (
                (seg_max - seg_min) * exclusion_percentage
            ).astype(int)

            center_min = seg_min + margin
            center_max = seg_max - margin

            if np.any(center_max <= center_min):
                center_min = seg_min
                center_max = seg_max

            slices = tuple(
                slice(
                    int(center_min[axis]),
                    int(center_max[axis]),
                )
                for axis in range(3)
            )

            prefix = (0,) * (seg.ndim - 3)
            mask_region = np.asarray(seg[prefix + slices])

            if not np.any(mask_region):
                empty_tile_ids.append(tile_id)

        ids_to_remove = {
            str(tile_id)
            for tile_id in empty_tile_ids
        }

        setup_ids = root.find(
            "./SequenceDescription/ImageLoader/SetupIds"
        )

        for definition in list(
            setup_ids.findall("SetupIdDefinition")
        ):
            if definition.findtext("NewId") in ids_to_remove:
                setup_ids.remove(definition)

        zgroups = root.find(
            "./SequenceDescription/ImageLoader/ImageLoader/zgroups"
        )

        if zgroups is not None:
            for zgroup in list(zgroups.findall("zgroup")):
                if zgroup.get("setup") in ids_to_remove:
                    zgroups.remove(zgroup)
        
        view_setups = root.find(
            "./SequenceDescription/ViewSetups"
        )

        for view_setup in list(
            view_setups.findall("ViewSetup")
        ):
            if view_setup.findtext("id") in ids_to_remove:
                view_setups.remove(view_setup)

        tile_attributes = view_setups.find(
            'Attributes[@name="tile"]'
        )

        for tile in list(tile_attributes.findall("Tile")):
            if tile.findtext("id") in ids_to_remove:
                tile_attributes.remove(tile)

        view_registrations = root.find(
            "./ViewRegistrations"
        )

        for registration in list(
            view_registrations.findall("ViewRegistration")
        ):
            if registration.get("setup") in ids_to_remove:
                view_registrations.remove(registration)

        view_interest_points = root.find(
            "./ViewInterestPoints"
        )

        if view_interest_points is not None:
            for entry in list(
                view_interest_points.findall(
                    "ViewInterestPointsFile"
                )
            ):
                if entry.get("setup") in ids_to_remove:
                    view_interest_points.remove(entry)

        ET.indent(tree, space="  ")

        if moving_split_xml_path.startswith("s3://"):
            with fsspec.open(
                moving_split_xml_path,
                "wb",
            ) as file:
                tree.write(
                    file,
                    encoding="UTF-8",
                    xml_declaration=True,
                )
        else:
            tree.write(
                moving_split_xml_path,
                encoding="UTF-8",
                xml_declaration=True,
            )

        print(
            f"Removed {len(empty_tile_ids)} empty moving tiles "
            f"from {len(tile_bboxes_zyx)} total tiles "
            f"using segmentation level {scale_level}"
        )

        return tile_bboxes_zyx

    def get_block_size_and_overlap_1d(self, size, num_blocks):
        if num_blocks == 1:
            return size, 0

        for block_size in range(size, 0, -1):
            total_overlap = (num_blocks * block_size) - size

            if total_overlap < 0:
                continue

            if total_overlap % (num_blocks - 1) != 0:
                continue

            overlap = total_overlap // (num_blocks - 1)

            if overlap <= block_size // 2:
                return block_size, overlap

        raise ValueError(f"Could not tile size={size} with num_blocks={num_blocks}")

    def get_tile_schedule(self, moving_image_shape, moving_image_voxel_size):
        shape_zyx = tuple(moving_image_shape[-3:])
        spacing_zyx = tuple(moving_image_voxel_size[-3:])

        schedule = []

        loop_index = 0
        while True:
            if loop_index == 0:
                grid_zyx = (1, 1, 1)
            else:
                xy = 2 ** (loop_index + 1)
                z = max(1, round(xy * shape_zyx[0] / shape_zyx[1]))
                grid_zyx = (z, xy, xy)

            block_size_zyx = []
            overlap_zyx = []

            for size, blocks in zip(shape_zyx, grid_zyx):
                block_size, overlap = self.get_block_size_and_overlap_1d(size, blocks)
                block_size_zyx.append(block_size)
                overlap_zyx.append(overlap)

            block_size_zyx = tuple(block_size_zyx)
            overlap_zyx = tuple(overlap_zyx)

            block_size_um_zyx = tuple(
                block_size_zyx[i] * spacing_zyx[i]
                for i in range(3)
            )

            if any(size_um < self.minimum_alignment_blocksize_physical_units for size_um in block_size_um_zyx):
                break

            schedule.append(grid_zyx)
            loop_index += 1

        return schedule

    def build_dataset_xml(self, image_root, xml_filename, xml_output_dir, setup_0_size_xyz, setup_0_voxel_size_xyz):
        root_path = image_root.rstrip("/")

        if root_path.startswith("s3://"):
            s3_path = root_path.removeprefix("s3://")
            s3_bucket, object_key = s3_path.split("/", 1)
            setup_0_path, setup_0_name = object_key.rsplit("/", 1)
        else:
            s3_bucket = None
            setup_0_path, setup_0_name = root_path.rsplit("/", 1)

        size_text = " ".join(map(str, setup_0_size_xyz))
        voxel_text = " ".join(map(str, setup_0_voxel_size_xyz))

        root = ET.Element("SpimData", version="0.2")
        ET.SubElement(root, "BasePath", type="relative").text = "."

        sequence = ET.SubElement(root, "SequenceDescription")

        loader = ET.SubElement(sequence, "ImageLoader", format="bdv.multimg.zarr", version="1.0")

        if s3_bucket is not None:
            ET.SubElement(loader, "s3bucket").text = s3_bucket

        ET.SubElement(loader, "zarr", type="absolute").text = setup_0_path

        zgroups = ET.SubElement(loader, "zgroups")
        zgroup = ET.SubElement(zgroups, "zgroup", setup="0", timepoint="0")
        ET.SubElement(zgroup, "path").text = setup_0_name

        view_setups = ET.SubElement(sequence, "ViewSetups")
        view_setup = ET.SubElement(view_setups, "ViewSetup")

        ET.SubElement(view_setup, "id").text = "0"
        ET.SubElement(view_setup, "size").text = size_text

        voxel_size = ET.SubElement(view_setup, "voxelSize")
        ET.SubElement(voxel_size, "unit").text = "micrometer"
        ET.SubElement(voxel_size, "size").text = voxel_text

        attributes = ET.SubElement(view_setup, "attributes")
        ET.SubElement(attributes, "illumination").text = "0"
        ET.SubElement(attributes, "channel").text = "0"
        ET.SubElement(attributes, "tile").text = "0"
        ET.SubElement(attributes, "angle").text = "0"

        for name, element_name in [
            ("illumination", "Illumination"),
            ("channel", "Channel"),
            ("tile", "Tile"),
            ("angle", "Angle"),
        ]:
            attribute_group = ET.SubElement(view_setups, "Attributes", name=name)
            attribute = ET.SubElement(attribute_group, element_name)
            ET.SubElement(attribute, "id").text = "0"
            ET.SubElement(attribute, "name").text = "0"

        timepoints = ET.SubElement(sequence, "Timepoints", type="range")
        ET.SubElement(timepoints, "first").text = "0"
        ET.SubElement(timepoints, "last").text = "0"
        ET.SubElement(sequence, "MissingViews")

        registrations = ET.SubElement(root, "ViewRegistrations")
        registration = ET.SubElement(registrations, "ViewRegistration", timepoint="0", setup="0")

        transform = ET.SubElement(registration, "ViewTransform", type="affine")
        ET.SubElement(transform, "Name").text = "calibration"

        voxel_x, voxel_y, voxel_z = setup_0_voxel_size_xyz
        ET.SubElement(transform, "affine").text = (
            f"{voxel_x} 0 0 0 "
            f"0 {voxel_y} 0 0 "
            f"0 0 {voxel_z} 0"
        )

        for section in [
            "ViewInterestPoints",
            "BoundingBoxes",
            "PointSpreadFunctions",
            "StitchingResults",
            "IntensityAdjustments",
        ]:
            ET.SubElement(root, section)

        xml_path = (
            f"{xml_output_dir.rstrip('/')}/{xml_filename}"
            if xml_output_dir.startswith("s3://")
            else os.path.join(xml_output_dir, xml_filename)
        )

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")

        if xml_path.startswith("s3://"):
            with fsspec.open(xml_path, "wb") as file:
                tree.write(file, encoding="UTF-8", xml_declaration=True)
        else:
            os.makedirs(xml_output_dir, exist_ok=True)
            tree.write(xml_path, encoding="UTF-8", xml_declaration=True)

        print(f"XML saved: {xml_path}")
        return xml_path

    def create_dataset_xml(self, image_root, xml_filename, xml_output_dir):
        root_path = image_root.rstrip("/")

        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=root_path, s3=s3, check=False)
        group = zarr.open_group(store, mode="r")

        base_image = group["0"]
        base_size_xyz = list(base_image.shape[-3:][::-1])

        base_metadata = next(
            dataset
            for dataset in group.attrs["multiscales"][0]["datasets"]
            if str(dataset["path"]).strip("/") == "0"
        )

        base_scale = next(
            transform
            for transform in base_metadata["coordinateTransformations"]
            if transform["type"] == "scale"
        )

        base_voxel_size_xyz = list(base_scale["scale"][-3:][::-1])

        return self.build_dataset_xml(image_root, xml_filename, xml_output_dir, base_size_xyz, 
                                      base_voxel_size_xyz)
    
    def create_tile_schedule(self, image_root):
        root = zarr.open(fsspec.get_mapper(image_root.rstrip("/")), mode="r")

        level = str(self.min_alignment_level)
        image_shape_zyx = tuple(root[level].shape[-3:])

        level_metadata = next(
            dataset
            for dataset in root.attrs["multiscales"][0]["datasets"]
            if str(dataset["path"]).strip("/") == level
        )

        scale_transform = next(
            transform
            for transform in level_metadata["coordinateTransformations"]
            if transform["type"] == "scale"
        )

        voxel_size_zyx = tuple(scale_transform["scale"][-3:])

        return self.get_tile_schedule(image_shape_zyx, voxel_size_zyx)

    def combine_detected_xmls(self, moving_detected_xml_path, fixed_detected_xml_path, output_xml_path):
        if moving_detected_xml_path.startswith("s3://"):
            with fsspec.open(moving_detected_xml_path, "rb") as file:
                moving_tree = ET.parse(file)
        else:
            moving_tree = ET.parse(moving_detected_xml_path)

        if fixed_detected_xml_path.startswith("s3://"):
            with fsspec.open(fixed_detected_xml_path, "rb") as file:
                fixed_root = ET.parse(file).getroot()
        else:
            fixed_root = ET.parse(fixed_detected_xml_path).getroot()

        moving_root = moving_tree.getroot()

        moving_view_setups = moving_root.find("./SequenceDescription/ViewSetups")
        fixed_view_setup = fixed_root.find("./SequenceDescription/ViewSetups/ViewSetup")

        existing_setup_ids = {
            setup.findtext("id")
            for setup in moving_view_setups.findall("ViewSetup")
        }

        if fixed_view_setup.findtext("id") not in existing_setup_ids:
            insert_index = next(
                (i for i, child in enumerate(moving_view_setups) if child.tag == "Attributes"),
                len(moving_view_setups),
            )
            moving_view_setups.insert(insert_index, copy.deepcopy(fixed_view_setup))

        moving_registrations = moving_root.find("./ViewRegistrations")
        existing_registration_ids = {
            (registration.get("timepoint"), registration.get("setup"))
            for registration in moving_registrations.findall("ViewRegistration")
        }

        for registration in fixed_root.findall("./ViewRegistrations/ViewRegistration"):
            key = (registration.get("timepoint"), registration.get("setup"))
            if key not in existing_registration_ids:
                moving_registrations.append(copy.deepcopy(registration))

        moving_interest_points = moving_root.find("./ViewInterestPoints")
        if moving_interest_points is None:
            moving_interest_points = ET.SubElement(moving_root, "ViewInterestPoints")

        existing_interest_point_ids = {
            (entry.get("timepoint"), entry.get("setup"), entry.get("label"))
            for entry in moving_interest_points.findall("ViewInterestPointsFile")
        }

        for entry in fixed_root.findall("./ViewInterestPoints/ViewInterestPointsFile"):
            key = (entry.get("timepoint"), entry.get("setup"), entry.get("label"))
            if key not in existing_interest_point_ids:
                moving_interest_points.append(copy.deepcopy(entry))

        ET.indent(moving_tree, space="  ")

        if output_xml_path.startswith("s3://"):
            with fsspec.open(output_xml_path, "wb") as file:
                moving_tree.write(file, encoding="UTF-8", xml_declaration=True)
        else:
            moving_tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)

        print(f"Combined detected XML: {output_xml_path}")
        return output_xml_path
