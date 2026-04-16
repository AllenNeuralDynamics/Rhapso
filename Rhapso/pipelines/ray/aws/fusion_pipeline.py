import yaml
import subprocess
import base64
import json
from pathlib import Path

with open("Rhapso/pipelines/ray/param/fusion/HCR_823476.yml", "r") as file:
    config = yaml.safe_load(file)

serialized_config = base64.b64encode(json.dumps(config).encode()).decode()

fusion_cmd = (
    "bash -lc \""
    "python3 - <<\\\"PY\\\"\n"
    "import json, base64\n"
    "from Rhapso.pipelines.ray.fusion import AffineFusion\n"
    f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
    "fusion = AffineFusion(\n"
    "    xml_path=cfg[\\\"xml_path_affine_fusion\\\"],\n"
    "    input_path=cfg[\\\"input_path_affine_fusion\\\"],\n"
    "    output_s3_path=cfg[\\\"output_s3_path_affine_fusion\\\"],\n"
    "    channel=cfg[\\\"channel\\\"],\n"
    "    default_chunk_size=cfg[\\\"default_chunk_size\\\"],\n"
    "    cpu_cell_size=cfg[\\\"cpu_cell_size\\\"],\n"
    ")\n"
    "fusion.execute_job()\n"
    "PY\n"
    "\""
)

multiscale_cmd = (
    "bash -lc \""
    "python3 - <<\\\"PY\\\"\n"
    "import json, base64\n"
    "from Rhapso.pipelines.ray.multiscale import MultiScale\n"
    f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
    "ms = MultiScale(\n"
    "    zarr_path=cfg[\\\"multiscale_zarr_path\\\"],\n"
    "    chunk_size=cfg[\\\"multiscale_chunk_size\\\"],\n"
    "    voxel_size=cfg[\\\"voxel_size\\\"],\n"
    "    n_lvls=cfg[\\\"n_lvls\\\"],\n"
    "    scale_factor=cfg[\\\"scale_factor\\\"],\n"
    "    target_block_size_mb=cfg[\\\"target_block_size_mb\\\"],\n"
    "    base_level=cfg[\\\"base_level\\\"],\n"
    ")\n"
    "ms.run()\n"
    "PY\n"
    "\""
)

prefix = (Path(__file__).resolve().parent / "config/dev").as_posix()
unified_yml = "fusion_cluster_sean.yml" 

def exec_on_cluster(name, yml, cmd, cwd):
    print(f"\n=== {name} ===")
    print("$", " ".join(["ray", "exec", yml, cmd]))
    subprocess.run(["ray", "exec", yml, cmd], check=True, cwd=cwd)

print("\n=== Start cluster ===")
print("$", " ".join(["ray", "up", unified_yml, "-y"]))
subprocess.run(["ray", "up", unified_yml, "-y"], check=True, cwd=prefix)

try:
    # exec_on_cluster("Affine Fusion", unified_yml, fusion_cmd, prefix)
    exec_on_cluster("Multiscale", unified_yml, multiscale_cmd, prefix)
    print("\n✅ Fusion + Multiscale pipeline complete.")

except subprocess.CalledProcessError as e:
    print(f"❌ Fusion pipeline error: {e}")
    raise

finally:
    print("\n=== Tear down cluster ===")
    print("$", " ".join(["ray", "down", unified_yml, "-y"]))
    subprocess.run(["ray", "down", unified_yml, "-y"], cwd=prefix)