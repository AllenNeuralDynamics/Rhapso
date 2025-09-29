"""
NOTE:
Codebase intended for GPU/CPU device.
No fallback to CPU written until required.
"""

import uuid
import time
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import yaml
import Rhapso.fusion.aind_cloud_fusion.fusion as fusion
import Rhapso.fusion.aind_cloud_fusion.input_output as input_output
import Rhapso.fusion.aind_cloud_fusion.script_utils as utils
import xml.etree.ElementTree as ET
import boto3
from io import BytesIO
import os
import multiprocessing as mp
import subprocess
import ray

def get_tile_zyx_resolution(input_xml_path: str) -> list[int]: 
    """
    Parse tile resolution to store in ome_ngff metadata
    """
    if input_xml_path.startswith('s3://'):
        # Handle S3 path
        s3 = boto3.resource('s3')
        bucket_name, key = input_xml_path[5:].split('/', 1)
        bucket = s3.Bucket(bucket_name)
        obj = bucket.Object(key)
        response = obj.get()
        file_stream = BytesIO(response['Body'].read())
        tree = ET.parse(file_stream)
    else:
        tree = ET.parse(input_xml_path)
    
    root = tree.getroot()

    res_xyz = root.find('SequenceDescription').find('ViewSetups').find('ViewSetup').find('voxelSize').find('size').text
    res_zyx = [float(num) for num in res_xyz.split(' ')[::-1]]
    
    return res_zyx

def cleanup_cluster(yml_path, cwd):
    """Clean up the Ray cluster and handle any errors gracefully"""
    print("\n=== Cleaning up cluster ===")
    print("$", " ".join(["ray", "down", yml_path, "-y"]))
    try:
        subprocess.run(["ray", "down", yml_path, "-y"], cwd=cwd, capture_output=False, text=True)
        print("✅ Cluster cleanup completed")
    except Exception as cleanup_error:
        print(f"⚠️  Cluster cleanup failed: {cleanup_error}")
        # Try alternative cleanup methods
        try:
            print("Trying alternative cleanup...")
            subprocess.run(["ray", "down", yml_path], cwd=cwd, capture_output=False, text=True)
        except:
            print("Alternative cleanup also failed - cluster may need manual cleanup")

def start_cluster(yml_path, cwd):
    """Start the Ray cluster"""
    print("\n=== Start cluster ===")
    print("$", " ".join(["ray", "up", yml_path, "-y"]))
    try:
        result = subprocess.run(["ray", "up", yml_path, "-y"], check=True, cwd=cwd, capture_output=False, text=True)
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
        cleanup_cluster(yml_path, cwd)
        raise

def execute_job(yml_path, xml_path, output_path, ray_config_path):
    # Get the directory containing the ray config file
    ray_config_dir = Path(ray_config_path).parent
    
    # Clean up any existing cluster first
    print("\n=== Clean up any existing cluster ===")
    print("$", " ".join(["ray", "down", Path(ray_config_path).name, "-y"]))
    try:
        subprocess.run(["ray", "down", Path(ray_config_path).name, "-y"], cwd=ray_config_dir, capture_output=False, text=True)
        print("✅ Cleanup completed (or no existing cluster)")
    except:
        print("ℹ️  No existing cluster to clean up")
    
    # Start the Ray cluster
    start_cluster(Path(ray_config_path).name, ray_config_dir)
    
    try:
        # Prep inputs
        configs = utils.read_config_yaml(yml_path)
        input_path = configs['input_path']
        output_s3_path = configs['output_path']
        channel = configs['channel']

        resolution_zyx = get_tile_zyx_resolution(xml_path)

        custom_chunksize = (1, 1, 128, 128, 128)
        custom_cpu_cell_size = (512, 256, 256)        
        # custom_chunksize = (1, 1, 3584, 1800, 3904)
        # custom_cpu_cell_size = (3584, 900, 1952) 

        output_params = input_output.OutputParameters(
            path=output_s3_path,
            resolution_zyx=resolution_zyx,
            chunk_size=custom_chunksize
        )
        blend_option = 'weighted_linear_blending'

        # Create the fusion command to run on the cluster (following alignment_pipeline.py pattern)
        fusion_cmd = (
            "bash -lc \""
            "python3 - <<\\\"PY\\\"\n"
            "import sys, os\n"
            "sys.path.append('/home/ubuntu')\n"
            "import Rhapso.fusion.aind_cloud_fusion.fusion as fusion\n"
            "import Rhapso.fusion.aind_cloud_fusion.input_output as input_output\n"
            "from datetime import datetime\n"
            "\n"
            "# Set environment variables\n"
            "os.environ[\\\"CUDA_VISIBLE_DEVICES\\\"] = \\\"\\\"\n"
            "os.environ[\\\"PYTORCH_CUDA_ALLOC_CONF\\\"] = \\\"max_split_size_mb:32\\\"\n"
            "# Set Ray serialization settings to handle large objects\n"
            "os.environ[\\\"RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE\\\"] = \\\"1\\\"\n"
            "os.environ[\\\"RAY_OBJECT_STORE_MEMORY\\\"] = \\\"10000000000\\\"\n"
            "\n"
            "# Define custom chunksize for large volumes\n"
            f"custom_chunksize = {custom_chunksize}\n"
            "# Workaround: proportional cell_size similar to working example to satisfy incorrect validation\n"
            f"custom_cpu_cell_size = {custom_cpu_cell_size}\n"
            "\n"
            "# Create output parameters\n"
            "output_params = input_output.OutputParameters(\n"
            f"    path=\\\"{output_s3_path}\\\",\n"
            f"    resolution_zyx={resolution_zyx},\n"
            "    chunk_size=custom_chunksize\n"
            ")\n"
            "\n"
            "print(f'Starting fusion at: {{datetime.now()}}')\n"
            f"print(f'Output fused zarr will be saved to: {output_s3_path}')\n"
            "\n"
            "# Run fusion with batch_size=150 and custom cell size\n"
            "fusion.run_fusion(\n"
            f"    \\\"{input_path}\\\",\n"
            f"    \\\"{xml_path}\\\",\n"
            f"    {channel},\n"
            "    output_params,\n"
            f"    \\\"{blend_option}\\\",\n"
            "    batch_size=150,\n"
            "    cpu_cell_size=custom_cpu_cell_size\n"
            ")\n"
            "\n"
            "print(f'Fusion completed at: {{datetime.now()}}')\n"
            f"print(f'Output fused zarr saved to: {output_s3_path}')\n"
            "PY\n"
            "\""
        )

        # Run fusion on the cluster using ray exec
        print(f'Starting fusion on cluster at: {datetime.now()}')
        print(f'Output fused zarr will be saved to: {output_s3_path}')
        
        # Run with timeout and better error handling
        try:
            result = subprocess.run(
                ["ray", "exec", Path(ray_config_path).name, fusion_cmd],
                cwd=ray_config_dir,
                capture_output=False,
                text=True,
                check=True,
                timeout=86400  # 24 hour timeout
            )
        except subprocess.TimeoutExpired:
            print("❌ Fusion timed out after 24 hours")
            cleanup_cluster(Path(ray_config_path).name, ray_config_dir)
            raise
        except subprocess.CalledProcessError as e:
            print(f"❌ Fusion failed with exit code {e.returncode}")
            cleanup_cluster(Path(ray_config_path).name, ray_config_dir)
            raise
        
        print(f'Fusion completed at: {datetime.now()}')
        print(f'Output fused zarr saved to: {output_s3_path}')
        
    except Exception as e:
        print(f"❌ Fusion error: {e}")
        # Clean up cluster on fusion error
        cleanup_cluster(Path(ray_config_path).name, ray_config_dir)
        raise
    
    finally:
        # Always try to clean up, even if everything succeeded
        cleanup_cluster(Path(ray_config_path).name, ray_config_dir)

    # Log 'done' file for next capsule in pipeline.
    # Unique log filename
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    unique_file_name = str(Path(output_path) / f"file_{timestamp}_{unique_id}.yml")

    log_content = {}
    log_content['in_path'] = output_params.path
    log_content['output_path'] = output_params.path.replace("fused_full_res", "fused")
    log_content['resolution_zyx'] = list(output_params.resolution_zyx)

    # Upload to S3 or write locally based on path
    if output_path.startswith('s3://'):
        # Upload to S3
        s3 = boto3.client('s3')
        bucket_name = output_path.split('/')[2]  # Extract bucket name from s3://bucket/path/
        s3_key = '/'.join(output_path.split('/')[3:]) + f"file_{timestamp}_{unique_id}.yml"
        
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=yaml.dump(log_content)
        )
        print(f"Log file uploaded to S3: s3://{bucket_name}/{s3_key}")
    else:
        # Write to local file
        with open(unique_file_name, "w") as file:
            yaml.dump(log_content, file)
        print(f"Log file written locally: {unique_file_name}")


if __name__ == '__main__':

    # Force CPU-only execution settings
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all GPUs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    print(f"Current multiprocessing start method: {mp.get_start_method(allow_none=False)}")
    print(f"Setting multiprocessing start method to 'forkserver': {mp.set_start_method('forkserver', force=True)}")
    print(f"New multiprocessing start method: {mp.get_start_method(allow_none=False)}")

    xml_path = "s3://martin-test-bucket/fusion/dataset.xml"
    yml_path = 's3://martin-test-bucket/fusion/worker_config.yml'
    output_path = 's3://martin-test-bucket/fusion/results/'
    ray_config_path = '/mnt/c/Users/marti/Documents/allen/repos/Rhapso-Fusion/Rhapso/pipelines/ray/aws/config/dev/fusion_cluster_martin.yml'

    print(f'{xml_path=}')
    print(f'{yml_path=}')
    print(f'{output_path=}')
    print(f'{ray_config_path=}')

    execute_job(yml_path,
                xml_path,
                output_path,
                ray_config_path)