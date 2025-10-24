"""
Runs fusion from config file generated from scheduler.
Manages full Ray cluster lifecycle (ray up/exec/down) on AWS.
"""

import os
import signal
import sys
import subprocess
from datetime import datetime
from pathlib import Path

from Rhapso.fusion.affine_fusion import script_utils as script_utils


# Global state for cleanup
ray_config_path = None
should_cleanup = False


def cleanup_cluster(yml_filename: str, cwd: Path):
    """Clean up the Ray cluster and handle any errors gracefully"""
    global should_cleanup
    if should_cleanup:
        return  # Already cleaned up
    
    print("\n=== Cleaning up cluster ===")
    print("$", " ".join(["ray", "down", yml_filename, "-y"]))
    try:
        subprocess.run(["ray", "down", yml_filename, "-y"], cwd=cwd, capture_output=False, text=True)
        print("✅ Cluster cleanup completed")
    except Exception as cleanup_error:
        print(f"⚠️  Cluster cleanup failed: {cleanup_error}")
        # Try alternative cleanup methods
        try:
            print("Trying alternative cleanup...")
            subprocess.run(["ray", "down", yml_filename], cwd=cwd, capture_output=False, text=True)
        except:
            print("Alternative cleanup also failed - cluster may need manual cleanup")
    
    should_cleanup = True


def start_cluster(yml_filename: str, cwd: Path):
    """Start the Ray cluster"""
    print("\n=== Start cluster ===")
    print("$", " ".join(["ray", "up", yml_filename, "-y"]))
    try:
        result = subprocess.run(["ray", "up", yml_filename, "-y"], check=True, cwd=cwd, capture_output=False, text=True)
        print("✅ Cluster started successfully")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Cluster startup failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        print("\n🔍 Debugging tips:")
        print("1. Check if the cluster name is unique (try changing cluster_name in the yml file)")
        print("2. Verify AWS credentials and permissions")
        print("3. Check if the S3 bucket and rhapso wheel file exist")
        print("4. Try running 'ray down' first to clean up any existing cluster")
        
        # Clean up failed cluster immediately
        cleanup_cluster(yml_filename, cwd)
        raise


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global ray_config_path
    print("\n\n⚠️  Interrupt received (Ctrl+C). Cleaning up...")
    if ray_config_path:
        ray_config_dir = Path(ray_config_path).parent
        yml_filename = Path(ray_config_path).name
        cleanup_cluster(yml_filename, ray_config_dir)
    print("✅ Cleanup completed. Exiting.")
    sys.exit(0)


def execute_job(yml_path: str, xml_path: str, cluster_config_path: str):
    """
    Execute fusion job with full Ray cluster lifecycle management.
    
    yml_path: Path to worker YAML config (local or S3)
    xml_path: Path to BigStitcher XML file (local or S3)
    cluster_config_path: Path to Ray cluster config YAML (e.g., fusion_cluster_martin.yml)
    """
    global ray_config_path
    ray_config_path = cluster_config_path
    
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"🚀 FUSION JOB STARTED: {start_time}")
    print(f"{'='*60}")
    
    # Get the directory containing the ray config file
    ray_config_dir = Path(cluster_config_path).parent
    yml_filename = Path(cluster_config_path).name
    
    try:
        # Clean up any existing cluster first
        print("\n=== Clean up any existing cluster ===")
        print("$", " ".join(["ray", "down", yml_filename, "-y"]))
        try:
            subprocess.run(["ray", "down", yml_filename, "-y"], cwd=ray_config_dir, capture_output=False, text=True)
            print("✅ Cleanup completed (or no existing cluster)")
        except:
            print("ℹ️  No existing cluster to clean up")
        
        # Start the Ray cluster
        start_cluster(yml_filename, ray_config_dir)
        
        # Parse information from worker yaml
        print(f"\n📋 Loading configuration from: {yml_path}")
        configs = script_utils.read_config_yaml(yml_path)
        input_path = configs['input_path']
        output_s3_path = configs['output_path']
        dataset_type = configs['dataset_type']
        channel = int(configs['channel'])
        worker_cells = configs['worker_cells']
        
        print(f"   Dataset type: {dataset_type}")
        print(f"   Channel: {channel}")
        print(f"   Input: {input_path}")
        print(f"   Output: {output_s3_path}")
        print(f"   Worker cells: {len(worker_cells)}")

        # Create the fusion command to run on the cluster
        # This command will be executed remotely via ray exec
        fusion_cmd = (
            "bash -lc \""
            "python3 - <<\\\"PY\\\"\n"
            "import sys, os\n"
            "sys.path.append('/home/ubuntu')\n"
            "\n"
            "# Set environment variables\n"
            "os.environ[\\\"CUDA_VISIBLE_DEVICES\\\"] = \\\"\\\"\n"
            "os.environ[\\\"PYTORCH_CUDA_ALLOC_CONF\\\"] = \\\"max_split_size_mb:32\\\"\n"
            "os.environ[\\\"RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE\\\"] = \\\"1\\\"\n"
            "os.environ[\\\"RAY_OBJECT_STORE_MEMORY\\\"] = \\\"10000000000\\\"\n"
            "\n"
            "from datetime import datetime\n"
            "from Rhapso.fusion.affine_fusion import blend, fusion, geometry, io, script_utils\n"
            "\n"
            "print(f'Starting fusion at: {{datetime.now()}}')\n"
            f"print(f'Loading configuration from: {yml_path}')\n"
            "\n"
            "# Parse worker config\n"
            f"configs = script_utils.read_config_yaml(\\\"{yml_path}\\\")\n"
            "input_path = configs['input_path']\n"
            "output_s3_path = configs['output_path']\n"
            "dataset_type = configs['dataset_type']\n"
            "channel = int(configs['channel'])\n"
            "worker_cells = [tuple(cell) for cell in configs['worker_cells']]\n"
            "\n"
            "print(f'Loading dataset: {{dataset_type}}, channel: {{channel}}')\n"
            "\n"
            "# Initialize dataset\n"
            f"if dataset_type == 'BigStitcherDataset':\n"
            f"    dataset = io.BigStitcherDataset(\\\"{xml_path}\\\", input_path, datastore=0)\n"
            f"elif dataset_type == 'BigStitcherDatasetChannel':\n"
            f"    dataset = io.BigStitcherDatasetChannel(\\\"{xml_path}\\\", input_path, channel, datastore=0)\n"
            "\n"
            "# Initialize output parameters\n"
            "output_params = io.OutputParameters(\n"
            "    path=output_s3_path,\n"
            "    chunksize=(1, 1, 128, 128, 128),\n"
            "    resolution_zyx=dataset.tile_resolution_zyx,\n"
            "    datastore=0\n"
            ")\n"
            "\n"
            "# Initialize runtime parameters\n"
            "runtime_params = io.RuntimeParameters(\n"
            "    option=2,\n"
            "    pool_size=int(os.environ.get('CO_CPUS', 1)),\n"
            "    worker_cells=worker_cells\n"
            ")\n"
            "\n"
            "# Set parameters\n"
            "cell_size = [128, 128, 128]\n"
            "post_reg_tfms = []\n"
            "\n"
            "print(f'Initializing fusion...')\n"
            "_, _, _, _, tile_aabbs, _, _ = fusion.initialize_fusion(\n"
            "    dataset, post_reg_tfms, output_params\n"
            ")\n"
            "\n"
            "# Initialize blending\n"
            "blending_module = blend.WeightedLinearBlending(tile_aabbs)\n"
            "\n"
            "print(f'Running fusion...')\n"
            "fusion.run_fusion(\n"
            "    dataset,\n"
            "    output_params,\n"
            "    runtime_params,\n"
            "    cell_size,\n"
            "    post_reg_tfms,\n"
            "    blending_module,\n"
            ")\n"
            "\n"
            "print(f'Fusion completed at: {{datetime.now()}}')\n"
            f"print(f'Output saved to: {{output_s3_path}}')\n"
            "PY\n"
            "\""
        )

        # Run fusion on the cluster using ray exec
        print(f'\n🔄 Starting fusion on cluster at: {datetime.now()}')
        print(f'   Output will be saved to: {output_s3_path}')
        
        # Run with timeout and better error handling
        try:
            result = subprocess.run(
                ["ray", "exec", yml_filename, fusion_cmd],
                cwd=ray_config_dir,
                capture_output=False,
                text=True,
                check=True,
                timeout=86400  # 24 hour timeout
            )
        except subprocess.TimeoutExpired:
            print("❌ Fusion timed out after 24 hours")
            cleanup_cluster(yml_filename, ray_config_dir)
            raise
        except subprocess.CalledProcessError as e:
            print(f"❌ Fusion failed with exit code {e.returncode}")
            cleanup_cluster(yml_filename, ray_config_dir)
            raise
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n{'='*60}")
        print(f"✅ FUSION JOB COMPLETED: {end_time}")
        print(f"⏱️  Duration: {duration}")
        print(f"📊 Output saved to: {output_s3_path}")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Job interrupted by user")
        raise
    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n{'='*60}")
        print(f"❌ FUSION JOB FAILED: {end_time}")
        print(f"⏱️  Duration before failure: {duration}")
        print(f"❌ Error: {e}")
        print(f"{'='*60}\n")
        raise
    finally:
        # Always try to clean up, even if everything succeeded
        cleanup_cluster(yml_filename, ray_config_dir)


if __name__ == '__main__':
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configuration paths
    WORKER_YML_PATH = "s3://martin-test-bucket/HCR_802704/yaml_configs/worker_0_ch488.yml"
    XML_PATH = "s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_tile_alignment/bigstitcher.xml"
    RAY_CLUSTER_CONFIG = 'Rhapso/pipelines/ray/aws/config/dev/fusion_cluster_martin.yml'

    print(f'{WORKER_YML_PATH=}')
    print(f'{XML_PATH=}')
    print(f'{RAY_CLUSTER_CONFIG=}')

    try:
        execute_job(WORKER_YML_PATH, XML_PATH, RAY_CLUSTER_CONFIG)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)
