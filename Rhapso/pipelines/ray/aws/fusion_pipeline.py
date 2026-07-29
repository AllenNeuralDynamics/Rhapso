import yaml
import subprocess
import base64
import json
from pathlib import Path

with open("Rhapso/pipelines/ray/param/fusion/exaSPIM_791116.yml", "r") as file:
    config = yaml.safe_load(file)

REMOTE_PYTHON = "/home/ubuntu/rhapso-py311/bin/python"

serialized_config = base64.b64encode(json.dumps(config).encode()).decode()

fusion_cmd = (
    "bash -lc \""
    f"{REMOTE_PYTHON} - <<\\\"PY\\\"\n"
    "import json, base64\n"
    "from Rhapso.pipelines.ray.affine_fusion import AffineFusion\n"
    f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
    "fusion = AffineFusion(\n"
    "    aligned_xml_path=cfg[\\\"aligned_xml_path\\\"],\n"
    "    zarr_input_prefix=cfg[\\\"zarr_input_prefix\\\"],\n"
    "    output_path=cfg[\\\"output_path\\\"],\n"
    "    block_size=cfg[\\\"block_size\\\"],\n"
    "    output_block_size=cfg[\\\"output_block_size\\\"],\n"
    "    intensity_range=cfg[\\\"intensity_range\\\"],\n"
    "    overlap_strategy=cfg[\\\"overlap_strategy\\\"],\n"
    "    output_zarr_version=cfg[\\\"output_zarr_version\\\"],\n"
    "    compressor_cname=cfg[\\\"compressor_cname\\\"],\n"
    "    compressor_clevel=cfg[\\\"compressor_clevel\\\"],\n"
    "    compressor_shuffle=cfg[\\\"compressor_shuffle\\\"],\n"
    ")\n"
    "fusion.run()\n"
    "PY\n"
    "\""
)

multiscale_cmd = (
    "bash -lc \""
    f"{REMOTE_PYTHON} - <<\\\"PY\\\"\n"
    "import json, base64\n"
    "from Rhapso.pipelines.ray.multiscale import MultiScale\n"
    f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
    "ms = MultiScale(\n"
    "    zarr_path=cfg[\\\"multiscale_zarr_path\\\"],\n"
    "    chunk_size=cfg[\\\"multiscale_chunk_size\\\"],\n"
    "    voxel_size=cfg[\\\"voxel_size\\\"],\n"
    "    n_lvls=cfg[\\\"n_lvls\\\"],\n"
    "    scale_factor=cfg[\\\"scale_factor\\\"],\n"
    "    base_level=cfg[\\\"base_level\\\"],\n"
    "    compressor_cname=cfg[\\\"compressor_cname\\\"],\n"
    "    compressor_clevel=cfg[\\\"compressor_clevel\\\"],\n"
    "    compressor_shuffle=cfg[\\\"compressor_shuffle\\\"],\n"
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
    exec_on_cluster("Affine Fusion", unified_yml, fusion_cmd, prefix)
    exec_on_cluster("Multiscale", unified_yml, multiscale_cmd, prefix)
    print("\n✅ Fusion + Multiscale pipeline complete.")

except subprocess.CalledProcessError as e:
    print(f"❌ Fusion pipeline error: {e}")
    raise

finally:
    print("\n=== Tear down cluster ===")
    print("$", " ".join(["ray", "down", unified_yml, "-y"]))
    subprocess.run(["ray", "down", unified_yml, "-y"], cwd=prefix)
    