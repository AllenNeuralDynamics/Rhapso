from Rhapso.pipelines.ray.solver import Solver
import yaml
import subprocess
import json
import base64, json
from pathlib import Path

with open("Rhapso/pipelines/ray/param/dev/zarr_s3_sean.yml", "r") as file:
    config = yaml.safe_load(file)

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
    "    max_spots=cfg[\\\"max_spots\\\"], median_filter=cfg[\\\"median_filter\\\"],\n"
    ")\n"
    "ipd.run()\n"
    "PY\n"
    "\""
)

# Rigid Matching run command
matching_cmd_rigid = (
    "bash -lc \""
    "python3 - <<\\\"PY\\\"\n"
    "import json, base64\n"
    "from Rhapso.pipelines.ray.interest_point_matching import InterestPointMatching\n"
    f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
    "ipm = InterestPointMatching(\n"
    "    xml_input_path=cfg[\\\"xml_file_path_matching_rigid\\\"],\n"
    "    n5_output_path=cfg[\\\"n5_matching_output_path\\\"],\n"
    "    input_type=cfg[\\\"input_type\\\"],\n"
    "    match_type=cfg[\\\"match_type_rigid\\\"],\n"
    "    num_neighbors=cfg[\\\"num_neighbors_rigid\\\"],\n"
    "    redundancy=cfg[\\\"redundancy_rigid\\\"],\n"
    "    significance=cfg[\\\"significance_rigid\\\"],\n"
    "    search_radius=cfg[\\\"search_radius_rigid\\\"],\n"
    "    num_required_neighbors=cfg[\\\"num_required_neighbors_rigid\\\"],\n"
    "    model_min_matches=cfg[\\\"model_min_matches_rigid\\\"],\n"
    "    inlier_factor=cfg[\\\"inlier_factor_rigid\\\"],\n"
    "    lambda_value=cfg[\\\"lambda_value_rigid\\\"],\n"
    "    num_iterations=cfg[\\\"num_iterations_rigid\\\"],\n"
    "    regularization_weight=cfg[\\\"regularization_weight_rigid\\\"],\n"
    "    image_file_prefix=cfg[\\\"image_file_prefix\\\"]\n"
    ")\n"
    "ipm.run()\n"
    "PY\n"
    "\""
)

# Affine matching run command
matching_cmd_affine = (
    "bash -lc \""
    "python3 - <<\\\"PY\\\"\n"
    "import json, base64\n"
    "from Rhapso.pipelines.ray.interest_point_matching import InterestPointMatching\n"
    f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
    "ipm = InterestPointMatching(\n"
    "    xml_input_path=cfg[\\\"xml_file_path_matching_affine\\\"],\n"
    "    n5_output_path=cfg[\\\"n5_matching_output_path\\\"],\n"
    "    input_type=cfg[\\\"input_type\\\"],\n"
    "    match_type=cfg[\\\"match_type_affine\\\"],\n"
    "    num_neighbors=cfg[\\\"num_neighbors_affine\\\"],\n"
    "    redundancy=cfg[\\\"redundancy_affine\\\"],\n"
    "    significance=cfg[\\\"significance_affine\\\"],\n"
    "    search_radius=cfg[\\\"search_radius_affine\\\"],\n"
    "    num_required_neighbors=cfg[\\\"num_required_neighbors_affine\\\"],\n"
    "    model_min_matches=cfg[\\\"model_min_matches_affine\\\"],\n"
    "    inlier_factor=cfg[\\\"inlier_factor_affine\\\"],\n"
    "    lambda_value=cfg[\\\"lambda_value_affine\\\"],\n"
    "    num_iterations=cfg[\\\"num_iterations_affine\\\"],\n"
    "    regularization_weight=cfg[\\\"regularization_weight_affine\\\"],\n"
    "    image_file_prefix=cfg[\\\"image_file_prefix\\\"]\n"
    ")\n"
    "ipm.run()\n"
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

# Split run command
split_cmd = (
    "bash -lc \""
    "python3 - <<\\\"PY\\\"\n"
    "import sys, json, base64\n"
    "from Rhapso.pipelines.ray.split_dataset import SplitDataset\n"
    f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
    "split = SplitDataset(\n"
    "    xml_file_path=cfg[\\\"xml_file_path_split\\\"],\n"
    "    xml_output_file_path=cfg[\\\"xml_output_file_path_split\\\"],\n"
    "    n5_path=cfg[\\\"n5_path_split\\\"], point_density=cfg[\\\"point_density\\\"], min_points=cfg[\\\"min_points\\\"],\n"
    "    max_points=cfg[\\\"max_points\\\"],\n"
    "    error=cfg[\\\"error\\\"],\n"
    "    exclude_radius=cfg[\\\"exclude_radius\\\"], target_image_size=cfg[\\\"target_image_size\\\"],\n"
    "    target_overlap=cfg[\\\"target_overlap\\\"],\n"
    ")\n"
    "split.run()\n"
    "PY\n"
    "\""
)

# Rigid solver run command
solver_rigid = Solver(
    xml_file_path_output=config['xml_file_path_output_rigid'],
    n5_input_path=config['n5_input_path'],
    xml_file_path=config['xml_file_path_solver_rigid'],
    run_type=config['run_type_solver_rigid'],   
    relative_threshold=config['relative_threshold'],
    absolute_threshold=config['absolute_threshold'],
    min_matches=config['min_matches'],
    damp=config['damp'],
    max_iterations=config['max_iterations'],
    max_allowed_error=config['max_allowed_error'],
    max_plateauwidth=config['max_plateauwidth'],
    metrics_output_path=config['metrics_output_path'],
    fixed_tile=config['fixed_tile']
)

# Affine solver run command
solver_affine = Solver(
    xml_file_path_output=config['xml_file_path_output_affine'],
    n5_input_path=config['n5_input_path'],
    xml_file_path=config['xml_file_path_solver_affine'],
    run_type=config['run_type_solver_affine'],  
    relative_threshold=config['relative_threshold'],
    absolute_threshold=config['absolute_threshold'],
    min_matches=config['min_matches'],
    damp=config['damp'],
    max_iterations=config['max_iterations'],
    max_allowed_error=config['max_allowed_error'],
    max_plateauwidth=config['max_plateauwidth'],
    metrics_output_path=config['metrics_output_path'],
    fixed_tile=config['fixed_tile']
)

# SOLVER SPLIT AFFINE
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
    fixed_tile=config['fixed_tile']
)

prefix = (Path(__file__).resolve().parent / "config/dev").as_posix()
unified_yml = "alignment_cluster_sean.yml"

def exec_on_cluster(name, yml, cmd, cwd):
    print(f"\n=== {name} ===")
    print("$", " ".join(["ray", "exec", yml, cmd]))
    subprocess.run(["ray", "exec", yml, cmd], check=True, cwd=cwd)

print("\n=== Start cluster ===")
print("$", " ".join(["ray", "up", unified_yml, "-y"]))
subprocess.run(["ray", "up", unified_yml, "-y"], check=True, cwd=prefix)

try:
    exec_on_cluster("Detection", unified_yml, detection_cmd, prefix)
    exec_on_cluster("Matching (rigid)", unified_yml, matching_cmd_rigid, prefix)
    solver_rigid.run()
    exec_on_cluster("Matching (affine)", unified_yml, matching_cmd_affine, prefix)
    solver_affine.run()
    exec_on_cluster("Split Dataset", unified_yml, split_cmd, prefix)
    exec_on_cluster("Matching (split_affine)", unified_yml, matching_cmd_split_affine, prefix)
    solver_split_affine.run()
    print("\n✅ Pipeline complete.")

except subprocess.CalledProcessError as e:
    print(f"❌ Pipeline error: {e}")
    raise

finally:
    print("\n=== Tear down cluster ===")
    print("$", " ".join(["ray", "down", unified_yml, "-y"]))
    subprocess.run(["ray", "down", unified_yml, "-y"], cwd=prefix)

print("\n✅ Pipeline complete.")