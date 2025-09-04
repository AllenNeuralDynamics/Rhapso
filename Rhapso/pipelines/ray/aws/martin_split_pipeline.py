"""
Martin Split Pipeline - Complete pipeline with detection, split-images, split-affine matching, and solver.

This pipeline combines:
1. Interest point detection
2. Image splitting (split-images)
3. Split-affine matching
4. Solver optimization

Run as module: `python3 -m Rhapso.pipelines.ray.aws.martin_split_pipeline`
"""

from Rhapso.pipelines.ray.solver import Solver
from Rhapso.image_split.split_datasets import main as split_images_main
import yaml
import subprocess
import json
import base64
import time
from pathlib import Path


def run_aws_s3_pipeline():
    """
    Sample method call for AWS S3 paths.
    This method runs the complete pipeline using AWS S3 input/output paths.
    """
    print("=== Running AWS S3 Pipeline ===")
    
    # Load configuration
    with open(Path("Rhapso/pipelines/ray/param/exaSPIM_686951_split.yml"), "r") as file:
        config = yaml.safe_load(file)
    
    serialized_config = base64.b64encode(json.dumps(config).encode()).decode()
    
    # Split Image Variables Configuration
    split_image_vars = {
        'xml_input': config['split_xml_input'],
        'xml_output': config['split_xml_output'], 
        'n5_output': config['split_n5_output'],
        'target_image_size_string': config['target_image_size_string'],
        'target_overlap_string': config['target_overlap_string'],
        'fake_interest_points': config['fake_interest_points'],
        'fip_exclusion_radius': config['fip_exclusion_radius'],
        'assign_illuminations': config['assign_illuminations'],
        'disable_optimization': config['disable_optimization'],
        'fip_density': config['fip_density'],
        'fip_min_num_points': config['fip_min_num_points'],
        'fip_max_num_points': config['fip_max_num_points'],
        'fip_error': config['fip_error']
    }
    
    # Detection run command
    detection_cmd = (
        "bash -lc \""
        "python3 - <<\\\"PY\\\"\n"
        "import sys, json, base64\n"
        "from Rhapso.pipelines.ray.interest_point_detection import InterestPointDetection\n"
        f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
        "ipd = InterestPointDetection(\n"
        "    dsxy=cfg[\\\"dsxy\\\"], dsz=cfg[\\\"dsz\\\"],\n"
        "    min_intensity=cfg[\\\"min_intensity\\\"], max_intensity=cfg[\\\"max_intensity\\\"],\n"
        "    sigma=cfg[\\\"sigma\\\"], threshold=cfg[\\\"threshold\\\"], file_type=cfg[\\\"file_type\\\"],\n"
        "    xml_file_path=cfg[\\\"xml_file_path_detection\\\"],\n"
        "    image_file_prefix=cfg[\\\"image_file_prefix\\\"],\n"
        "    xml_output_file_path=cfg[\\\"xml_output_file_path\\\"], n5_output_file_prefix=cfg[\\\"n5_output_file_prefix\\\"],\n"
        "    combine_distance=cfg[\\\"combine_distance\\\"],\n"
        "    chunks_per_bound=cfg[\\\"chunks_per_bound\\\"], run_type=cfg[\\\"detection_run_type\\\"],\n"
        "    max_spots=cfg[\\\"max_spots\\\"], median_filter=cfg[\\\"median_filter\\\"]\n"
        ")\n"
        "ipd.run()\n"
        "PY\n"
        "\""
    )
    
    # Split affine matching run command
    matching_cmd_split_affine = (
        "bash -lc \""
        "python3 - <<\\\"PY\\\"\n"
        "import json, base64\n"
        "from Rhapso.pipelines.ray.interest_point_matching import InterestPointMatching\n"
        f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
        "ipm = InterestPointMatching(\n"
        "    xml_input_path=cfg[\\\"xml_file_path_matching_split_affine\\\"],\n"
        "    n5_output_path=cfg[\\\"n5_matching_output_path\\\"],\n"
        "    input_type=cfg[\\\"input_type\\\"],\n"
        "    match_type=cfg[\\\"match_type_split_affine\\\"],\n"
        "    num_neighbors=cfg[\\\"num_neighbors_split_affine\\\"],\n"
        "    redundancy=cfg[\\\"redundancy_split_affine\\\"],\n"
        "    significance=cfg[\\\"significance_split_affine\\\"],\n"
        "    search_radius=cfg[\\\"search_radius_split_affine\\\"],\n"
        "    num_required_neighbors=cfg[\\\"num_required_neighbors_split_affine\\\"],\n"
        "    model_min_matches=cfg[\\\"model_min_matches_split_affine\\\"],\n"
        "    inlier_factor=cfg[\\\"inlier_factor_split_affine\\\"],\n"
        "    lambda_value=cfg[\\\"lambda_value_split_affine\\\"],\n"
        "    num_iterations=cfg[\\\"num_iterations_split_affine\\\"],\n"
        "    regularization_weight=cfg[\\\"regularization_weight_split_affine\\\"],\n"
        "    image_file_prefix=cfg[\\\"image_file_prefix\\\"]\n"
        ")\n"
        "ipm.run()\n"
        "PY\n"
        "\""
    )
    
    # Split affine solver
    solver_split_affine = Solver(
        xml_file_path_output=config['xml_file_path_output_split_affine'],
        n5_input_path=config['n5_input_path'],
        xml_file_path=config['xml_file_path_solver_split_affine'],
        run_type=config['run_type_solver_split_affine'],
        relative_threshold=config['relative_threshold'],
        absolute_threshold=config['absolute_threshold'],
        min_matches=config['min_matches'],
        damp=config['damp'],
        max_iterations=config['max_iterations'],
        max_allowed_error=config['max_allowed_error'],
        max_plateauwidth=config['max_plateauwidth'],
        metrics_output_path=config['metrics_output_path'],
    )
    
    prefix = (Path(__file__).resolve().parent / "config").as_posix()
    unified_yml = "alignment_cluster_martin.yml"
    
    def exec_on_cluster(name, yml, cmd, cwd):
        print(f"\n=== {name} ===")
        print("$", " ".join(["ray", "exec", yml, cmd]))
        subprocess.run(["ray", "exec", yml, cmd], check=True, cwd=cwd)
    
    print("\n=== Start cluster ===")
    print("$", " ".join(["ray", "up", unified_yml, "-y"]))
    subprocess.run(["ray", "up", unified_yml, "-y"], check=True, cwd=prefix)
    
    try:
        # Step 1: Detection
        exec_on_cluster("Detection", unified_yml, detection_cmd, prefix)
        
        # Step 2: Split Images
        print("\n=== Split Images ===")
        print("Running image split pipeline...")
        split_images_main(
            xml_input=split_image_vars['xml_input'],
            xml_output=split_image_vars['xml_output'],
            n5_output=split_image_vars['n5_output'],
            target_image_size_string=split_image_vars['target_image_size_string'],
            target_overlap_string=split_image_vars['target_overlap_string'],
            fake_interest_points=split_image_vars['fake_interest_points'],
            fip_exclusion_radius=split_image_vars['fip_exclusion_radius'],
            assign_illuminations=split_image_vars['assign_illuminations'],
            disable_optimization=split_image_vars['disable_optimization'],
            fip_density=split_image_vars['fip_density'],
            fip_min_num_points=split_image_vars['fip_min_num_points'],
            fip_max_num_points=split_image_vars['fip_max_num_points'],
            fip_error=split_image_vars['fip_error']
        )
        print("Image split pipeline completed.")
        
        # Step 3: Split Affine Matching
        exec_on_cluster("Matching (split_affine)", unified_yml, matching_cmd_split_affine, prefix)
        
        # Step 4: Solver
        solver_split_affine.run()
        
        print("\n✅ AWS S3 Pipeline complete.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline error: {e}")
        raise
    finally:
        print("\n=== Tear down cluster ===")
        print("$", " ".join(["ray", "down", unified_yml, "-y"]))
        subprocess.run(["ray", "down", unified_yml, "-y"], cwd=prefix)


def run_local_paths_pipeline():
    """
    Sample method call for local file paths.
    This method runs the complete pipeline using local input/output paths.
    """
    print("=== Running Local Paths Pipeline ===")
    
    # Load configuration
    with open(Path("Rhapso/pipelines/ray/param/exaSPIM_686951_split.yml"), "r") as file:
        config = yaml.safe_load(file)
    
    serialized_config = base64.b64encode(json.dumps(config).encode()).decode()
    
    # Local Split Image Variables Configuration
    local_split_image_vars = {
        'xml_input': "/mnt/c/Users/marti/Documents/allen/data/exaSPIM_686951 EXAMPLE/ip_affine_alignment/bigstitcher_affine.xml",
        'xml_output': "/mnt/c/Users/marti/Documents/allen/data/exaSPIM_686951 EXAMPLE/results/bigstitcher_affine_split.xml",
        'n5_output': "/mnt/c/Users/marti/Documents/allen/data/exaSPIM_686951 EXAMPLE/results/interestpoints.n5",
        'target_image_size_string': config['target_image_size_string'],
        'target_overlap_string': config['target_overlap_string'],
        'fake_interest_points': config['fake_interest_points'],
        'fip_exclusion_radius': config['fip_exclusion_radius'],
        'assign_illuminations': config['assign_illuminations'],
        'disable_optimization': config['disable_optimization'],
        'fip_density': config['fip_density'],
        'fip_min_num_points': config['fip_min_num_points'],
        'fip_max_num_points': config['fip_max_num_points'],
        'fip_error': config['fip_error']
    }
    
    # Detection run command
    detection_cmd = (
        "bash -lc \""
        "python3 - <<\\\"PY\\\"\n"
        "import sys, json, base64\n"
        "from Rhapso.pipelines.ray.interest_point_detection import InterestPointDetection\n"
        f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
        "ipd = InterestPointDetection(\n"
        "    dsxy=cfg[\\\"dsxy\\\"], dsz=cfg[\\\"dsz\\\"],\n"
        "    min_intensity=cfg[\\\"min_intensity\\\"], max_intensity=cfg[\\\"max_intensity\\\"],\n"
        "    sigma=cfg[\\\"sigma\\\"], threshold=cfg[\\\"threshold\\\"], file_type=cfg[\\\"file_type\\\"],\n"
        "    xml_file_path=cfg[\\\"xml_file_path_detection\\\"],\n"
        "    image_file_prefix=cfg[\\\"image_file_prefix\\\"],\n"
        "    xml_output_file_path=cfg[\\\"xml_output_file_path\\\"], n5_output_file_prefix=cfg[\\\"n5_output_file_prefix\\\"],\n"
        "    combine_distance=cfg[\\\"combine_distance\\\"],\n"
        "    chunks_per_bound=cfg[\\\"chunks_per_bound\\\"], run_type=cfg[\\\"detection_run_type\\\"],\n"
        "    max_spots=cfg[\\\"max_spots\\\"], median_filter=cfg[\\\"median_filter\\\"]\n"
        ")\n"
        "ipd.run()\n"
        "PY\n"
        "\""
    )
    
    # Split affine matching run command
    matching_cmd_split_affine = (
        "bash -lc \""
        "python3 - <<\\\"PY\\\"\n"
        "import json, base64\n"
        "from Rhapso.pipelines.ray.interest_point_matching import InterestPointMatching\n"
        f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
        "ipm = InterestPointMatching(\n"
        "    xml_input_path=cfg[\\\"xml_file_path_matching_split_affine\\\"],\n"
        "    n5_output_path=cfg[\\\"n5_matching_output_path\\\"],\n"
        "    input_type=cfg[\\\"input_type\\\"],\n"
        "    match_type=cfg[\\\"match_type_split_affine\\\"],\n"
        "    num_neighbors=cfg[\\\"num_neighbors_split_affine\\\"],\n"
        "    redundancy=cfg[\\\"redundancy_split_affine\\\"],\n"
        "    significance=cfg[\\\"significance_split_affine\\\"],\n"
        "    search_radius=cfg[\\\"search_radius_split_affine\\\"],\n"
        "    num_required_neighbors=cfg[\\\"num_required_neighbors_split_affine\\\"],\n"
        "    model_min_matches=cfg[\\\"model_min_matches_split_affine\\\"],\n"
        "    inlier_factor=cfg[\\\"inlier_factor_split_affine\\\"],\n"
        "    lambda_value=cfg[\\\"lambda_value_split_affine\\\"],\n"
        "    num_iterations=cfg[\\\"num_iterations_split_affine\\\"],\n"
        "    regularization_weight=cfg[\\\"regularization_weight_split_affine\\\"],\n"
        "    image_file_prefix=cfg[\\\"image_file_prefix\\\"]\n"
        ")\n"
        "ipm.run()\n"
        "PY\n"
        "\""
    )
    
    # Split affine solver
    solver_split_affine = Solver(
        xml_file_path_output=config['xml_file_path_output_split_affine'],
        n5_input_path=config['n5_input_path'],
        xml_file_path=config['xml_file_path_solver_split_affine'],
        run_type=config['run_type_solver_split_affine'],
        relative_threshold=config['relative_threshold'],
        absolute_threshold=config['absolute_threshold'],
        min_matches=config['min_matches'],
        damp=config['damp'],
        max_iterations=config['max_iterations'],
        max_allowed_error=config['max_allowed_error'],
        max_plateauwidth=config['max_plateauwidth'],
        metrics_output_path=config['metrics_output_path'],
    )
    
    prefix = (Path(__file__).resolve().parent / "config").as_posix()
    unified_yml = "alignment_cluster_martin.yml"
    
    def exec_on_cluster(name, yml, cmd, cwd):
        print(f"\n=== {name} ===")
        print("$", " ".join(["ray", "exec", yml, cmd]))
        subprocess.run(["ray", "exec", yml, cmd], check=True, cwd=cwd)
    
    print("\n=== Start cluster ===")
    print("$", " ".join(["ray", "up", unified_yml, "-y"]))
    subprocess.run(["ray", "up", unified_yml, "-y"], check=True, cwd=prefix)
    
    try:
        # Step 1: Detection
        exec_on_cluster("Detection", unified_yml, detection_cmd, prefix)
        
        # Step 2: Split Images
        print("\n=== Split Images ===")
        print("Running image split pipeline...")
        split_images_main(
            xml_input=local_split_image_vars['xml_input'],
            xml_output=local_split_image_vars['xml_output'],
            n5_output=local_split_image_vars['n5_output'],
            target_image_size_string=local_split_image_vars['target_image_size_string'],
            target_overlap_string=local_split_image_vars['target_overlap_string'],
            fake_interest_points=local_split_image_vars['fake_interest_points'],
            fip_exclusion_radius=local_split_image_vars['fip_exclusion_radius'],
            assign_illuminations=local_split_image_vars['assign_illuminations'],
            disable_optimization=local_split_image_vars['disable_optimization'],
            fip_density=local_split_image_vars['fip_density'],
            fip_min_num_points=local_split_image_vars['fip_min_num_points'],
            fip_max_num_points=local_split_image_vars['fip_max_num_points'],
            fip_error=local_split_image_vars['fip_error']
        )
        print("Image split pipeline completed.")
        
        # Step 3: Split Affine Matching
        exec_on_cluster("Matching (split_affine)", unified_yml, matching_cmd_split_affine, prefix)
        
        # Step 4: Solver
        solver_split_affine.run()
        
        print("\n✅ Local Paths Pipeline complete.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline error: {e}")
        raise
    finally:
        print("\n=== Tear down cluster ===")
        print("$", " ".join(["ray", "down", unified_yml, "-y"]))
        subprocess.run(["ray", "down", unified_yml, "-y"], cwd=prefix)


if __name__ == "__main__":
    run_aws_s3_pipeline()
    # run_local_paths_pipeline()
else:
    print("Martin Split Pipeline module imported.")
