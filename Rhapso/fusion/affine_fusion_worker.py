"""
Runs fusion from config file generated
from dispim or exaspim scheduler.
Manages full Ray cluster lifecycle (ray up/exec/down) on AWS.
"""

import signal
import sys
import subprocess
from pathlib import Path

from Rhapso.fusion.affine_fusion import blend as blend
from Rhapso.fusion.affine_fusion import fusion as fusion
from Rhapso.fusion.affine_fusion import geometry as geometry
from Rhapso.fusion.affine_fusion import io as io
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
        try:
            print("Trying alternative cleanup...")
            subprocess.run(["ray", "down", yml_filename], cwd=cwd, capture_output=False, text=True)
        except:
            print("Alternative cleanup also failed - cluster may need manual cleanup")
    
    should_cleanup = True


def cleanup_existing_cluster(yml_filename: str, cwd: Path):
    """Clean up any existing cluster before starting a new one"""
    print("\n=== Clean up any existing cluster ===")
    print("$", " ".join(["ray", "down", yml_filename, "-y"]))
    try:
        subprocess.run(["ray", "down", yml_filename, "-y"], cwd=cwd, capture_output=False, text=True)
        print("✅ Cleanup completed (or no existing cluster)")
    except:
        print("ℹ️  No existing cluster to clean up")


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


def execute_job(xml_path: str, image_data_input_path: str, output_s3_path: str, dataset_type: str, channel: int, ray_cluster_config_path: str):
    """
    Execute fusion job with full Ray cluster lifecycle management.
    
    xml_path: Path to BigStitcher XML file (local or S3)
    image_data_input_path: Path to input image data (local or S3)
    output_s3_path: Path to output location (local or S3)
    dataset_type: Type of dataset ('BigStitcherDataset' or 'BigStitcherDatasetChannel')
    channel: Channel number (int or None)
    ray_cluster_config_path: Path to Ray cluster config YAML (e.g., fusion_cluster_martin.yml)
    """
    global ray_config_path
    ray_config_path = ray_cluster_config_path
    
    # Get the directory containing the ray config file
    ray_config_dir = Path(ray_cluster_config_path).parent
    yml_filename = Path(ray_cluster_config_path).name
    
    try:
        # Clean up any existing cluster first
        cleanup_existing_cluster(yml_filename, ray_config_dir)
        
        # Start the Ray cluster
        start_cluster(yml_filename, ray_config_dir)
        
        # Application Parameters
        cell_size = [640, 256, 256]    
        chunksize = (1, 1, 640, 256, 256)   
        
        # Reconstruct objects on cluster and run fusion.run_fusion()
        channel_arg = f"channel={channel}" if channel is not None else "channel=None"
        
        # Create the fusion command to run on the cluster
        fusion_cmd = (
            "bash -lc \""
            "python3 - <<\\\"PY\\\"\n"
            "import sys\n"
            "sys.path.append('/home/ubuntu')\n"
            "\n"
            "from Rhapso.fusion.affine_fusion import blend, fusion, geometry, io\n"
            "\n"
            "# Reconstruct objects on cluster\n"
            f"dataset_type = \\\"{dataset_type}\\\"\n"
            f"{channel_arg}\n"
            f"xml_path = \\\"{xml_path}\\\"\n"
            f"image_data_input_path = \\\"{image_data_input_path}\\\"\n"
            f"output_s3_path = \\\"{output_s3_path}\\\"\n"
            f"cell_size = {cell_size}\n"
            f"chunksize = {chunksize}\n"
            "\n"
            "if dataset_type == 'BigStitcherDataset':\n"
            "    DATASET = io.BigStitcherDataset(xml_path, image_data_input_path, datastore=0)\n"
            "elif dataset_type == 'BigStitcherDatasetChannel':\n"
            "    DATASET = io.BigStitcherDatasetChannel(xml_path, image_data_input_path, channel, datastore=0)\n"
            "\n"
            "OUTPUT_PARAMS = io.OutputParameters(\n"
            "    path=output_s3_path,\n"
            "    chunksize=chunksize,\n"
            "    resolution_zyx=DATASET.tile_resolution_zyx,\n"
            "    datastore=0\n"
            ")\n"
            "\n"
            "CELL_SIZE = cell_size\n"
            "POST_REG_TFMS = []\n"
            "\n"
            "_, _, _, _, tile_aabbs, _, _ = fusion.initialize_fusion(\n"
            "    DATASET, POST_REG_TFMS, OUTPUT_PARAMS\n"
            ")\n"
            "\n"
            "BLENDING_MODULE = blend.WeightedLinearBlending(tile_aabbs)\n"
            "\n"
            "fusion.run_fusion(\n"
            "    DATASET,\n"
            "    OUTPUT_PARAMS,\n"
            "    CELL_SIZE,\n"
            "    POST_REG_TFMS,\n"
            "    BLENDING_MODULE,\n"
            ")\n"
            "PY\n"
            "\""
        )

        # Run fusion on the cluster using ray exec
        print(f'\n🔄 Starting fusion.run_fusion() on cluster')
        print(f'   Output will be saved to: {output_s3_path}')
        
        try:
            result = subprocess.run(
                ["ray", "exec", yml_filename, fusion_cmd],
                cwd=ray_config_dir,
                capture_output=False,
                text=True,
                check=True,
                timeout = 8 * 60 * 60  # 8 hour timeout
            )
        except subprocess.TimeoutExpired:
            print("❌ Fusion timed out after 8 hours")
            cleanup_cluster(yml_filename, ray_config_dir)
            raise
        except subprocess.CalledProcessError as e:
            print(f"❌ Fusion failed with exit code {e.returncode}")
            cleanup_cluster(yml_filename, ray_config_dir)
            raise
        
        print(f"\n{'='*60}")
        print(f"✅ FUSION JOB COMPLETED")
        print(f"📊 Output saved to: {output_s3_path}")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Job interrupted by user")
        raise
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ FUSION JOB FAILED")
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

    xml_path = "s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_tile_alignment/bigstitcher.xml"
    image_data_input_path = "s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_radial_correction/"
    output_s3_path = "s3://martin-test-bucket/output7/channel_488.zarr"
    dataset_type = "BigStitcherDataset"
    channel = None                      # list channel num (int) if fusing a specific channel from an xml of multiple channels
    ray_cluster_config_path = 'Rhapso/pipelines/ray/aws/config/dev/fusion_cluster_martin.yml'

    print(f'{xml_path=}')
    print(f'{image_data_input_path=}')
    print(f'{output_s3_path=}')
    print(f'{dataset_type=}')
    print(f'{channel=}')
    print(f'{ray_cluster_config_path=}')

    try:
        execute_job(xml_path, image_data_input_path, output_s3_path, dataset_type, channel, ray_cluster_config_path)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)