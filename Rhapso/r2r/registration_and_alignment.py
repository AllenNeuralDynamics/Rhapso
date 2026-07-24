import base64
import copy
import os
import shlex
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
import fsspec
import yaml
import zarr
from Rhapso.pipelines.ray.solver import Solver

REMOTE_PYTHON = "/home/ubuntu/rhapso-py311/bin/python"
CLUSTER_DIR = (Path(__file__).resolve().parents[1] / "pipelines/ray/aws/config/dev").as_posix()
CLUSTER_YML = "alignment_cluster_sean.yml"

class RegistrationAndAlignment:
    def __init__(self, fixed_image_multiscale_root, moving_image_multiscale_root, ip_registration_config_file):
        self.fixed_image_multiscale_root = fixed_image_multiscale_root
        self.moving_image_multiscale_root = moving_image_multiscale_root
        self.cluster_started = False

        with fsspec.open(ip_registration_config_file, "rt") as file:
            self.config = yaml.safe_load(file)

    def start_cluster(self):
        if self.cluster_started:
            return

        print("\n=== Start Ray cluster ===")
        subprocess.run(["ray", "up", CLUSTER_YML, "-y"], check=True, cwd=CLUSTER_DIR)
        self.cluster_started = True

    def stop_cluster(self):
        if not self.cluster_started:
            return

        print("\n=== Tear down Ray cluster ===")
        subprocess.run(["ray", "down", CLUSTER_YML, "-y"], cwd=CLUSTER_DIR)
        self.cluster_started = False

    def run_on_cluster(self, name, script):
        self.start_cluster()

        encoded = base64.b64encode(script.encode()).decode()
        command = f"printf %s {shlex.quote(encoded)} | base64 -d | {shlex.quote(REMOTE_PYTHON)}"

        print(f"\n=== {name} ===")
        subprocess.run(["ray", "exec", CLUSTER_YML, command], check=True, cwd=CLUSTER_DIR)

    def get_round_param(self, parameter, round_number):
        if round_number not in {1, 2, 3, 4}:
            raise ValueError(f"Round number must be 1 through 4, got {round_number}")

        key = f"{parameter}_round_{round_number}"

        if key not in self.config:
            raise KeyError(f"Missing registration parameter: {key}")

        return self.config[key]

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

    def detect_interest_points(self, input_xml_path, image_root, output_xml_path, point_store_dir, image_type, round_number):
        if image_type not in {"fixed", "moving"}:
            raise ValueError(f"image_type must be 'fixed' or 'moving', got '{image_type}'")

        image_prefix = image_root.rstrip("/").rsplit("/", 1)[0] + "/"
        point_store_dir = point_store_dir.rstrip("/") + "/"

        script = textwrap.dedent(f"""
            from Rhapso.pipelines.ray.interest_point_detection import InterestPointDetection

            detection = InterestPointDetection(
                dsxy={self.get_round_param("dsxy", round_number)!r}, dsz={self.get_round_param("dsz", round_number)!r},
                min_intensity={self.get_round_param(f"min_intensity_{image_type}", round_number)!r},
                max_intensity={self.get_round_param(f"max_intensity_{image_type}", round_number)!r},
                sigma={self.get_round_param("sigma", round_number)!r}, threshold={self.get_round_param("threshold", round_number)!r},
                file_type={self.get_round_param("file_type", round_number)!r}, xml_file_path={input_xml_path!r},
                image_file_prefix={image_prefix!r}, xml_output_file_path={output_xml_path!r},
                n5_output_file_prefix={point_store_dir!r}, combine_distance={self.get_round_param("combine_distance", round_number)!r},
                chunks_per_bound={self.get_round_param("chunks_per_bound", round_number)!r},
                run_type={self.get_round_param("run_type", round_number)!r}, max_spots={self.get_round_param("max_spots", round_number)!r},
                median_filter={self.get_round_param("median_filter", round_number)!r},
                overlap_only={self.get_round_param("overlap_only", round_number)!r},
            )

            detection.run()
        """).strip()

        self.run_on_cluster(f"Round {round_number} {image_type} detection", script)
        return output_xml_path

    def split_moving_dataset(self, input_xml_path, output_xml_path, block_grid_zyx):
        fixed_root = zarr.open(fsspec.get_mapper(self.fixed_image_multiscale_root.rstrip("/")), mode="r")
        moving_root = zarr.open(fsspec.get_mapper(self.moving_image_multiscale_root.rstrip("/")), mode="r")

        fixed_shape_zyx = tuple(fixed_root["0"].shape[-3:])
        moving_shape_zyx = tuple(moving_root["0"].shape[-3:])
        max_shape_zyx = tuple(max(fixed_shape_zyx[i], moving_shape_zyx[i]) for i in range(3))

        tile_size_zyx = []
        overlap_zyx = []

        for size, blocks in zip(max_shape_zyx, block_grid_zyx):
            block_size, overlap = self.get_block_size_and_overlap_1d(size, blocks)
            tile_size_zyx.append(block_size)
            overlap_zyx.append(overlap)

        tile_size_zyx = tuple(tile_size_zyx)
        overlap_zyx = tuple(overlap_zyx)

        print(f"Fixed shape ZYX: {fixed_shape_zyx}")
        print(f"Moving shape ZYX: {moving_shape_zyx}")
        print(f"Split grid ZYX: {block_grid_zyx}")
        print(f"Tile size ZYX: {tile_size_zyx}")
        print(f"Overlap ZYX: {overlap_zyx}")

        target_image_size = tuple(reversed(tile_size_zyx))
        target_overlap = tuple(reversed(overlap_zyx))

        script = textwrap.dedent(f"""
            from Rhapso.pipelines.ray.split_dataset import SplitDataset

            split_dataset = SplitDataset(
                xml_file_path={input_xml_path!r}, xml_output_file_path={output_xml_path!r},
                n5_path={self.config["n5_path_split"]!r}, point_density={self.config["point_density"]!r},
                min_points={self.config["min_points"]!r}, max_points={self.config["max_points"]!r},
                error={self.config["error"]!r}, exclude_radius={self.config["exclude_radius"]!r},
                target_image_size={target_image_size!r}, target_overlap={target_overlap!r},
            )

            split_dataset.run()
        """).strip()

        self.run_on_cluster("Split moving dataset", script)

        if output_xml_path.startswith("s3://"):
            with fsspec.open(output_xml_path, "rb") as file:
                tree = ET.parse(file)
        else:
            tree = ET.parse(output_xml_path)

        root = tree.getroot()
        zgroups = root.find("./SequenceDescription/ImageLoader/ImageLoader/zgroups")
        setup_ids = root.find("./SequenceDescription/ImageLoader/SetupIds")

        if zgroups is None:
            raise ValueError("Split XML is missing the nested Zarr zgroups")
        if setup_ids is None:
            raise ValueError("Split XML is missing SetupIds")

        source_zgroups = {}

        for zgroup in zgroups.findall("zgroup"):
            setup_id = int(zgroup.get("setup"))
            timepoint = int(zgroup.get("timepoint") or zgroup.get("tp") or 0)
            source_zgroups[(timepoint, setup_id)] = copy.deepcopy(zgroup)

        definitions = [
            (int(definition.findtext("NewId")), int(definition.findtext("OldId")))
            for definition in setup_ids.findall("SetupIdDefinition")
        ]

        for zgroup in list(zgroups.findall("zgroup")):
            zgroups.remove(zgroup)

        for new_id, old_id in definitions:
            matching_sources = [
                (timepoint, source_zgroup)
                for (timepoint, setup_id), source_zgroup in source_zgroups.items()
                if setup_id == old_id
            ]

            if not matching_sources:
                raise ValueError(f"No Zarr path found for split OldId {old_id}")

            for timepoint, source_zgroup in matching_sources:
                split_zgroup = copy.deepcopy(source_zgroup)
                split_zgroup.set("setup", str(new_id))

                if "tp" in split_zgroup.attrib:
                    split_zgroup.set("tp", str(timepoint))
                else:
                    split_zgroup.set("timepoint", str(timepoint))

                zgroups.append(split_zgroup)

        ET.indent(tree, space="  ")

        if output_xml_path.startswith("s3://"):
            with fsspec.open(output_xml_path, "wb") as file:
                tree.write(file, encoding="UTF-8", xml_declaration=True)
        else:
            tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)

        print(f"Expanded Zarr paths for {len(definitions)} split setups")
        return output_xml_path, tile_size_zyx, overlap_zyx

    def align_interest_points(self, combined_xml_path, fixed_setup_id, loop_output_dir, point_store_path, round_number):
        point_store_path = point_store_path.rstrip("/") + "/"
        registered_xml_path = os.path.join(loop_output_dir, "ip_registered.xml")

        script = textwrap.dedent(f"""
            from Rhapso.pipelines.ray.interest_point_matching import InterestPointMatching

            matching = InterestPointMatching(
                xml_input_path={combined_xml_path!r}, n5_output_path={point_store_path!r},
                input_type={self.get_round_param("input_type", round_number)!r},
                match_type={self.get_round_param("match_type", round_number)!r},
                num_neighbors={self.get_round_param("num_neighbors", round_number)!r},
                redundancy={self.get_round_param("redundancy", round_number)!r},
                significance={self.get_round_param("significance", round_number)!r},
                search_radius={self.get_round_param("search_radius", round_number)!r},
                num_required_neighbors={self.get_round_param("num_required_neighbors", round_number)!r},
                ransac_sample_size={self.get_round_param("ransac_sample_size", round_number)!r},
                model_min_inliers={self.get_round_param("model_min_inliers", round_number)!r},
                inlier_threshold={self.get_round_param("inlier_threshold", round_number)!r},
                min_inlier_ratio={self.get_round_param("min_inlier_ratio", round_number)!r},
                num_iterations={self.get_round_param("num_iterations", round_number)!r},
                regularization_weight={self.get_round_param("regularization_weight_matching", round_number)!r},
                image_file_prefix={self.moving_image_multiscale_root!r},
            )

            matching.run()
        """).strip()

        self.run_on_cluster(f"Round {round_number} matching", script)

        metrics_output_path = os.path.join(loop_output_dir, "metrics", "metrics.json")

        if not metrics_output_path.startswith("s3://"):
            os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

        solver = Solver(
            xml_file_path_output=registered_xml_path, n5_input_path=point_store_path, xml_file_path=combined_xml_path,
            run_type=self.get_round_param("run_type_solver", round_number),
            relative_threshold=self.get_round_param("relative_threshold", round_number),
            absolute_threshold=self.get_round_param("absolute_threshold", round_number),
            max_cleanup_rounds=self.get_round_param("max_cleanup_rounds", round_number),
            min_matches=self.get_round_param("min_matches", round_number), damp=self.get_round_param("damp", round_number),
            regularization_weight=self.get_round_param("regularization_weight_solver", round_number),
            max_iterations=self.get_round_param("max_iterations_solver", round_number),
            max_allowed_error=self.get_round_param("max_allowed_error", round_number),
            max_plateauwidth=self.get_round_param("max_plateauwidth", round_number),
            metrics_output_path=metrics_output_path, fixed_tile=f"timepoint: 0, setup: {fixed_setup_id}",
        )

        solver.run()
        return registered_xml_path
