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
    
    # Split images run command
    split_images_cmd = (
        "bash -lc \""
        "python3 - <<\\\"PY\\\"\n"
        "import json, base64\n"
        "from Rhapso.pipelines.ray.split_images import SplitImages\n"
        f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
        "si = SplitImages(\n"
        "    xml_input=cfg[\\\"split_xml_input\\\"],\n"
        "    xml_output=cfg[\\\"split_xml_output\\\"],\n"
        "    n5_output=cfg[\\\"split_n5_output\\\"],\n"
        "    target_image_size_string=cfg[\\\"target_image_size_string\\\"],\n"
        "    target_overlap_string=cfg[\\\"target_overlap_string\\\"],\n"
        "    fake_interest_points=cfg[\\\"fake_interest_points\\\"],\n"
        "    fip_exclusion_radius=cfg[\\\"fip_exclusion_radius\\\"],\n"
        "    assign_illuminations=cfg[\\\"assign_illuminations\\\"],\n"
        "    disable_optimization=cfg[\\\"disable_optimization\\\"],\n"
        "    fip_density=cfg[\\\"fip_density\\\"],\n"
        "    fip_min_num_points=cfg[\\\"fip_min_num_points\\\"],\n"
        "    fip_max_num_points=cfg[\\\"fip_max_num_points\\\"],\n"
        "    fip_error=cfg[\\\"fip_error\\\"]\n"
        ")\n"
        "si.run()\n"
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
        start_time = time.time()
        print(f"\n=== {name} ===")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "exec", yml, cmd]))
        subprocess.run(["ray", "exec", yml, cmd], check=True, cwd=cwd)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")
    
    print("\n=== Start cluster ===")
    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("$", " ".join(["ray", "up", unified_yml, "-y"]))
    subprocess.run(["ray", "up", unified_yml, "-y"], check=True, cwd=prefix)
    end_time = time.time()
    duration = end_time - start_time
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Duration: {duration:.2f} seconds")
    
    try:
        # Step 1: Detection
        exec_on_cluster("Detection", unified_yml, detection_cmd, prefix)
        
        # Step 2: Split Images (using Ray)
        exec_on_cluster("Split Images", unified_yml, split_images_cmd, prefix)
        
        # Step 3: Split Affine Matching
        exec_on_cluster("Matching (split_affine)", unified_yml, matching_cmd_split_affine, prefix)
        
        # Step 4: Solver
        start_time = time.time()
        print(f"\n=== Solver ===")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        solver_split_affine.run()
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")
        
        print("\n✅ AWS S3 Pipeline complete.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline error: {e}")
        raise
    finally:
        print("\n=== Tear down cluster ===")
        start_time = time.time()
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "down", unified_yml, "-y"]))
        subprocess.run(["ray", "down", unified_yml, "-y"], cwd=prefix)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")


def run_local_paths_pipeline():
    """
    Run only the split images step using local file paths from the YAML configuration.
    This method loads the local paths from the YAML file and runs only the split images step.
    """
    print("=== Running Local Split Images Only ===")
    
    # Load configuration
    with open(Path("Rhapso/pipelines/ray/param/exaSPIM_686951_split.yml"), "r") as file:
        config = yaml.safe_load(file)
    
    # Extract split images configuration from YAML
    split_config = {
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
    
    # Serialize the split configuration
    serialized_config = base64.b64encode(json.dumps(split_config).encode()).decode()
    
    # Split images run command using local paths from YAML
    split_images_cmd = (
        "bash -lc \""
        "python3 - <<\\\"PY\\\"\n"
        "import json, base64\n"
        "from Rhapso.pipelines.ray.split_images import SplitImages\n"
        f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
        "si = SplitImages(\n"
        "    xml_input=cfg[\\\"xml_input\\\"],\n"
        "    xml_output=cfg[\\\"xml_output\\\"],\n"
        "    n5_output=cfg[\\\"n5_output\\\"],\n"
        "    target_image_size_string=cfg[\\\"target_image_size_string\\\"],\n"
        "    target_overlap_string=cfg[\\\"target_overlap_string\\\"],\n"
        "    fake_interest_points=cfg[\\\"fake_interest_points\\\"],\n"
        "    fip_exclusion_radius=cfg[\\\"fip_exclusion_radius\\\"],\n"
        "    assign_illuminations=cfg[\\\"assign_illuminations\\\"],\n"
        "    disable_optimization=cfg[\\\"disable_optimization\\\"],\n"
        "    fip_density=cfg[\\\"fip_density\\\"],\n"
        "    fip_min_num_points=cfg[\\\"fip_min_num_points\\\"],\n"
        "    fip_max_num_points=cfg[\\\"fip_max_num_points\\\"],\n"
        "    fip_error=cfg[\\\"fip_error\\\"]\n"
        ")\n"
        "si.run()\n"
        "PY\n"
        "\""
    )
    
    prefix = (Path(__file__).resolve().parent / "config").as_posix()
    unified_yml = "alignment_cluster_martin.yml"
    
    def exec_on_cluster(name, yml, cmd, cwd):
        start_time = time.time()
        print(f"\n=== {name} ===")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "exec", yml, cmd]))
        subprocess.run(["ray", "exec", yml, cmd], check=True, cwd=cwd)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")
    
    print("\n=== Start cluster (with clear config) ===")
    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("$", " ".join(["ray", "up", unified_yml, "--no-config-cache", "-y"]))
    subprocess.run(["ray", "up", unified_yml, "--no-config-cache", "-y"], check=True, cwd=prefix)
    end_time = time.time()
    duration = end_time - start_time
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Duration: {duration:.2f} seconds")
    
    try:
        # Only run Split Images step
        exec_on_cluster("Split Images (Local Paths)", unified_yml, split_images_cmd, prefix)
        
        print("\n✅ Local Split Images Pipeline complete.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline error: {e}")
        raise
    finally:
        print("\n=== Tear down cluster (with clear config) ===")
        start_time = time.time()
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "down", unified_yml, "-y"]))
        subprocess.run(["ray", "down", unified_yml, "-y"], cwd=prefix)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")


def just_run_split_aws():
    """
    Run only the split images step using AWS S3 paths.
    This method uses aind-open-data as input and martin-test-bucket as output.
    """
    print("=== Running AWS S3 Split Images Only ===")
    
    # Load configuration
    with open(Path("Rhapso/pipelines/ray/param/exaSPIM_686951_split.yml"), "r") as file:
        config = yaml.safe_load(file)
    
    # Create AWS S3 split images configuration (hardcoded AWS S3 paths)
    aws_split_config = {
        'xml_input': 's3://aind-open-data/exaSPIM_686951_2025-02-25_09-45-02_alignment_2025-06-12_19-58-52/ip_affine_alignment/bigstitcher_affine.xml',  # Use original input XML
        'xml_output': 's3://martin-test-bucket/exaSPIM_686951_split/split_images/bigstitcher_affine_split.xml',
        'n5_output': 's3://aind-open-data/exaSPIM_686951_2025-02-25_09-45-02_alignment_2025-06-12_19-58-52/ip_affine_alignment/interestpoints.n5',  # Use original N5 input
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
    
    # Serialize the split configuration
    serialized_config = base64.b64encode(json.dumps(aws_split_config).encode()).decode()
    
    # Split images run command using AWS S3 paths
    split_images_cmd = (
        "bash -lc \""
        "python3 - <<\\\"PY\\\"\n"
        "import json, base64\n"
        "from Rhapso.pipelines.ray.split_images import SplitImages\n"
        f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
        "si = SplitImages(\n"
        "    xml_input=cfg[\\\"xml_input\\\"],\n"
        "    xml_output=cfg[\\\"xml_output\\\"],\n"
        "    n5_output=cfg[\\\"n5_output\\\"],\n"
        "    target_image_size_string=cfg[\\\"target_image_size_string\\\"],\n"
        "    target_overlap_string=cfg[\\\"target_overlap_string\\\"],\n"
        "    fake_interest_points=cfg[\\\"fake_interest_points\\\"],\n"
        "    fip_exclusion_radius=cfg[\\\"fip_exclusion_radius\\\"],\n"
        "    assign_illuminations=cfg[\\\"assign_illuminations\\\"],\n"
        "    disable_optimization=cfg[\\\"disable_optimization\\\"],\n"
        "    fip_density=cfg[\\\"fip_density\\\"],\n"
        "    fip_min_num_points=cfg[\\\"fip_min_num_points\\\"],\n"
        "    fip_max_num_points=cfg[\\\"fip_max_num_points\\\"],\n"
        "    fip_error=cfg[\\\"fip_error\\\"]\n"
        ")\n"
        "si.run()\n"
        "PY\n"
        "\""
    )
    
    prefix = (Path(__file__).resolve().parent / "config").as_posix()
    unified_yml = "alignment_cluster_martin.yml"
    
    def exec_on_cluster(name, yml, cmd, cwd):
        start_time = time.time()
        print(f"\n=== {name} ===")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "exec", yml, cmd]))
        subprocess.run(["ray", "exec", yml, cmd], check=True, cwd=cwd)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")
    
    print("\n=== Start cluster (with clear config) ===")
    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("$", " ".join(["ray", "up", unified_yml, "--no-config-cache", "-y"]))
    subprocess.run(["ray", "up", unified_yml, "--no-config-cache", "-y"], check=True, cwd=prefix)
    end_time = time.time()
    duration = end_time - start_time
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Duration: {duration:.2f} seconds")
    
    try:
        # Only run Split Images step with AWS S3 paths
        exec_on_cluster("Split Images (AWS S3)", unified_yml, split_images_cmd, prefix)
        
        print("\n✅ AWS S3 Split Images Pipeline complete.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline error: {e}")
        raise
    finally:
        print("\n=== Tear down cluster (with clear config) ===")
        start_time = time.time()
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "down", unified_yml, "-y"]))
        subprocess.run(["ray", "down", unified_yml, "-y"], cwd=prefix)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")


def detection():
    """
    Run only the detection step and output to s3://martin-test-bucket/detection
    """
    print("=== Running Detection Only ===")
    
    # Load configuration
    with open(Path("Rhapso/pipelines/ray/param/exaSPIM_686951_split.yml"), "r") as file:
        config = yaml.safe_load(file)
    
    # Modify config to output detection results to martin-test-bucket/detection
    config['xml_output_file_path'] = 's3://martin-test-bucket/detection/bigstitcher_affine.xml'
    config['n5_output_file_prefix'] = 's3://martin-test-bucket/detection/'
    
    serialized_config = base64.b64encode(json.dumps(config).encode()).decode()
    
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
    
    prefix = (Path(__file__).resolve().parent / "config").as_posix()
    unified_yml = "alignment_cluster_martin.yml"
    
    def exec_on_cluster(name, yml, cmd, cwd):
        start_time = time.time()
        print(f"\n=== {name} ===")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "exec", yml, cmd]))
        subprocess.run(["ray", "exec", yml, cmd], check=True, cwd=cwd)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")
    
    print("\n=== Start cluster ===")
    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("$", " ".join(["ray", "up", unified_yml, "-y"]))
    subprocess.run(["ray", "up", unified_yml, "-y"], check=True, cwd=prefix)
    end_time = time.time()
    duration = end_time - start_time
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Duration: {duration:.2f} seconds")
    
    try:
        # Run Detection only
        exec_on_cluster("Detection", unified_yml, detection_cmd, prefix)
        print("\n✅ Detection complete. Output saved to s3://martin-test-bucket/detection")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Detection error: {e}")
        raise
    finally:
        print("\n=== Tear down cluster ===")
        start_time = time.time()
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "down", unified_yml, "-y"]))
        subprocess.run(["ray", "down", unified_yml, "-y"], cwd=prefix)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")


def split():
    """
    Run only the split step using detection output as input
    """
    print("=== Running Split Only ===")
    
    # Load configuration
    with open(Path("Rhapso/pipelines/ray/param/exaSPIM_686951_split.yml"), "r") as file:
        config = yaml.safe_load(file)
    
    # Create split configuration using detection output as input
    split_config = {
        'xml_input': 's3://martin-test-bucket/detection/bigstitcher_affine.xml',  # Use detection output as input
        'xml_output': 's3://martin-test-bucket/detection/bigstitcher_affine_split.xml',
        'n5_output': 's3://martin-test-bucket/detection/interestpoints.n5',  # Use detection output as input
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
    
    # Serialize the split configuration
    serialized_config = base64.b64encode(json.dumps(split_config).encode()).decode()
    
    # Split images run command
    split_images_cmd = (
        "bash -lc \""
        "python3 - <<\\\"PY\\\"\n"
        "import json, base64\n"
        "from Rhapso.pipelines.ray.split_images import SplitImages\n"
        f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
        "si = SplitImages(\n"
        "    xml_input=cfg[\\\"xml_input\\\"],\n"
        "    xml_output=cfg[\\\"xml_output\\\"],\n"
        "    n5_output=cfg[\\\"n5_output\\\"],\n"
        "    target_image_size_string=cfg[\\\"target_image_size_string\\\"],\n"
        "    target_overlap_string=cfg[\\\"target_overlap_string\\\"],\n"
        "    fake_interest_points=cfg[\\\"fake_interest_points\\\"],\n"
        "    fip_exclusion_radius=cfg[\\\"fip_exclusion_radius\\\"],\n"
        "    assign_illuminations=cfg[\\\"assign_illuminations\\\"],\n"
        "    disable_optimization=cfg[\\\"disable_optimization\\\"],\n"
        "    fip_density=cfg[\\\"fip_density\\\"],\n"
        "    fip_min_num_points=cfg[\\\"fip_min_num_points\\\"],\n"
        "    fip_max_num_points=cfg[\\\"fip_max_num_points\\\"],\n"
        "    fip_error=cfg[\\\"fip_error\\\"]\n"
        ")\n"
        "si.run()\n"
        "PY\n"
        "\""
    )
    
    prefix = (Path(__file__).resolve().parent / "config").as_posix()
    unified_yml = "alignment_cluster_martin.yml"
    
    def exec_on_cluster(name, yml, cmd, cwd):
        start_time = time.time()
        print(f"\n=== {name} ===")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "exec", yml, cmd]))
        subprocess.run(["ray", "exec", yml, cmd], check=True, cwd=cwd)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")
    
    print("\n=== Start cluster ===")
    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("$", " ".join(["ray", "up", unified_yml, "-y"]))
    subprocess.run(["ray", "up", unified_yml, "-y"], check=True, cwd=prefix)
    end_time = time.time()
    duration = end_time - start_time
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Duration: {duration:.2f} seconds")
    
    try:
        # Run Split Images only
        exec_on_cluster("Split Images", unified_yml, split_images_cmd, prefix)
        print("\n✅ Split complete.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Split error: {e}")
        raise
    finally:
        print("\n=== Tear down cluster ===")
        start_time = time.time()
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "down", unified_yml, "-y"]))
        subprocess.run(["ray", "down", unified_yml, "-y"], cwd=prefix)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")


def detect_and_split():
    """
    Run detection followed by split in sequence, keeping all paths the same
    """
    print("=== Running Detection and Split Pipeline ===")
    
    # Load configuration
    with open(Path("Rhapso/pipelines/ray/param/exaSPIM_686951_split.yml"), "r") as file:
        config = yaml.safe_load(file)
    
    # Modify config to output detection results to martin-test-bucket/detection
    config['xml_output_file_path'] = 's3://martin-test-bucket/detection/bigstitcher_affine.xml'
    config['n5_output_file_prefix'] = 's3://martin-test-bucket/detection/'
    
    serialized_config = base64.b64encode(json.dumps(config).encode()).decode()
    
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
    
    # Create split configuration using detection output as input
    split_config = {
        'xml_input': 's3://martin-test-bucket/detection/bigstitcher_affine.xml',  # Use detection output as input
        'xml_output': 's3://martin-test-bucket/detection/bigstitcher_affine_split.xml',
        'n5_output': 's3://martin-test-bucket/detection/interestpoints.n5',  # Use detection output as input
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
    
    # Serialize the split configuration
    split_serialized_config = base64.b64encode(json.dumps(split_config).encode()).decode()
    
    # Split images run command
    split_images_cmd = (
        "bash -lc \""
        "python3 - <<\\\"PY\\\"\n"
        "import json, base64\n"
        "from Rhapso.pipelines.ray.split_images import SplitImages\n"
        f"cfg = json.loads(base64.b64decode(\\\"{split_serialized_config}\\\").decode())\n"
        "si = SplitImages(\n"
        "    xml_input=cfg[\\\"xml_input\\\"],\n"
        "    xml_output=cfg[\\\"xml_output\\\"],\n"
        "    n5_output=cfg[\\\"n5_output\\\"],\n"
        "    target_image_size_string=cfg[\\\"target_image_size_string\\\"],\n"
        "    target_overlap_string=cfg[\\\"target_overlap_string\\\"],\n"
        "    fake_interest_points=cfg[\\\"fake_interest_points\\\"],\n"
        "    fip_exclusion_radius=cfg[\\\"fip_exclusion_radius\\\"],\n"
        "    assign_illuminations=cfg[\\\"assign_illuminations\\\"],\n"
        "    disable_optimization=cfg[\\\"disable_optimization\\\"],\n"
        "    fip_density=cfg[\\\"fip_density\\\"],\n"
        "    fip_min_num_points=cfg[\\\"fip_min_num_points\\\"],\n"
        "    fip_max_num_points=cfg[\\\"fip_max_num_points\\\"],\n"
        "    fip_error=cfg[\\\"fip_error\\\"]\n"
        ")\n"
        "si.run()\n"
        "PY\n"
        "\""
    )
    
    prefix = (Path(__file__).resolve().parent / "config").as_posix()
    unified_yml = "alignment_cluster_martin.yml"
    
    def exec_on_cluster(name, yml, cmd, cwd):
        start_time = time.time()
        print(f"\n=== {name} ===")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "exec", yml, cmd]))
        subprocess.run(["ray", "exec", yml, cmd], check=True, cwd=cwd)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")
    
    print("\n=== Start cluster ===")
    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("$", " ".join(["ray", "up", unified_yml, "-y"]))
    subprocess.run(["ray", "up", unified_yml, "-y"], check=True, cwd=prefix)
    end_time = time.time()
    duration = end_time - start_time
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Duration: {duration:.2f} seconds")
    
    try:
        # Step 1: Run Detection
        exec_on_cluster("Detection", unified_yml, detection_cmd, prefix)
        print("\n✅ Detection complete. Output saved to s3://martin-test-bucket/detection")
        
        # Step 2: Run Split Images (using detection output as input)
        exec_on_cluster("Split Images", unified_yml, split_images_cmd, prefix)
        print("\n✅ Split complete.")
        
        print("\n✅ Detection and Split Pipeline complete!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline error: {e}")
        raise
    finally:
        print("\n=== Tear down cluster ===")
        start_time = time.time()
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print("$", " ".join(["ray", "down", unified_yml, "-y"]))
        subprocess.run(["ray", "down", unified_yml, "-y"], cwd=prefix)
        end_time = time.time()
        duration = end_time - start_time
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    #detection()
    #split()
    detect_and_split()
    #just_run_split_aws()
    #run_aws_s3_pipeline()
    #run_local_paths_pipeline()
else:
    print("Martin Split Pipeline module imported.")
