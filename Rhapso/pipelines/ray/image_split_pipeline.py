"""
Ray-distributed image split pipeline for Rhapso.

This pipeline automatically detects whether input/output paths are S3 URLs or local paths
and chooses the appropriate execution method:
- S3 paths: Uses AWS Ray cluster with ray exec commands
- Local paths: Uses local Ray initialization

The pipeline uses Ray to parallelize the N5 file creation step for fake interest points,
significantly improving performance when processing large numbers of image tiles.

Usage:
    python image_split_pipeline.py

Configuration:
    Edit the configuration variables at the top of this file to customize
    input/output paths and processing parameters.
"""

import os
import subprocess
import sys
import json
import base64
from pathlib import Path
from Rhapso.image_split.split_datasets import main, main_with_ray

# ============================================================================ 
# Configuration
# ============================================================================

# AWS S3 Input/Output Paths
# XML_INPUT = "s3://martin-test-bucket/split-images-test/bigstitcher_affine.xml"
# XML_OUTPUT = "s3://martin-test-bucket/split-images-test/bigstitcher_affine_split.xml"
# N5_OUTPUT = "s3://martin-test-bucket/split-images-test/interestpoints.n5"

# Local Input/Output Paths (commented out)
XML_INPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/bigstitcher_affine.xml"
XML_OUTPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/bigstitcher_affine_split.xml"
N5_OUTPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/interestpoints.n5"

# Image Splitting Parameters
TARGET_IMAGE_SIZE_STRING = "7000,7000,4000"
TARGET_OVERLAP_STRING = "128,128,128"
ASSIGN_ILLUMINATIONS = False
DISABLE_OPTIMIZATION = False

# Fake Interest Points Configuration
FAKE_INTEREST_POINTS = True
FIP_DENSITY = 100.0
FIP_MIN_NUM_POINTS = 20
FIP_MAX_NUM_POINTS = 500
FIP_EXCLUSION_RADIUS = 200.0
FIP_ERROR = 0.5


def is_s3_path(path):
    """Check if a path is an S3 URL."""
    return path.startswith("s3://")


def run_aws_cluster_pipeline():
    """Run the pipeline on AWS Ray cluster."""
    print("Detected S3 paths - using AWS Ray cluster execution...")
    
    # Create configuration dictionary
    config = {
        "xml_input": XML_INPUT,
        "xml_output": XML_OUTPUT,
        "n5_output": N5_OUTPUT,
        "target_image_size_string": TARGET_IMAGE_SIZE_STRING,
        "target_overlap_string": TARGET_OVERLAP_STRING,
        "fake_interest_points": FAKE_INTEREST_POINTS,
        "fip_density": FIP_DENSITY,
        "fip_min_num_points": FIP_MIN_NUM_POINTS,
        "fip_max_num_points": FIP_MAX_NUM_POINTS,
        "fip_exclusion_radius": FIP_EXCLUSION_RADIUS,
        "assign_illuminations": ASSIGN_ILLUMINATIONS,
        "disable_optimization": DISABLE_OPTIMIZATION,
        "fip_error": FIP_ERROR
    }
    
    serialized_config = base64.b64encode(json.dumps(config).encode()).decode()
    
    # Image split run command
    image_split_cmd = (
        "bash -lc \""
        "python3 - <<\\\"PY\\\"\n"
        "import sys, json, base64\n"
        "from Rhapso.image_split.split_datasets import main_with_ray\n"
        f"cfg = json.loads(base64.b64decode(\\\"{serialized_config}\\\").decode())\n"
        "main_with_ray(\n"
        "    xml_input=cfg[\\\"xml_input\\\"],\n"
        "    xml_output=cfg[\\\"xml_output\\\"],\n"
        "    n5_output=cfg[\\\"n5_output\\\"],\n"
        "    target_image_size_string=cfg[\\\"target_image_size_string\\\"],\n"
        "    target_overlap_string=cfg[\\\"target_overlap_string\\\"],\n"
        "    fake_interest_points=cfg[\\\"fake_interest_points\\\"],\n"
        "    fip_density=cfg[\\\"fip_density\\\"],\n"
        "    fip_min_num_points=cfg[\\\"fip_min_num_points\\\"],\n"
        "    fip_max_num_points=cfg[\\\"fip_max_num_points\\\"],\n"
        "    fip_exclusion_radius=cfg[\\\"fip_exclusion_radius\\\"],\n"
        "    assign_illuminations=cfg[\\\"assign_illuminations\\\"],\n"
        "    disable_optimization=cfg[\\\"disable_optimization\\\"],\n"
        "    fip_error=cfg[\\\"fip_error\\\"]\n"
        ")\n"
        "PY\n"
        "\""
    )
    
    # Get cluster config path
    script_dir = Path(__file__).resolve().parent
    config_dir = script_dir / "aws" / "config"
    cluster_yml = "alignment_cluster_martin.yml"
    
    def exec_on_cluster(name, yml, cmd, cwd):
        print(f"\n=== {name} ===")
        print("$", " ".join(["ray", "exec", yml, cmd]))
        subprocess.run(["ray", "exec", yml, cmd], check=True, cwd=cwd)
    
    print("\n=== Start cluster ===")
    print("$", " ".join(["ray", "up", cluster_yml, "-y"]))
    subprocess.run(["ray", "up", cluster_yml, "-y"], check=True, cwd=config_dir)
    
    try:
        exec_on_cluster("Image Split", cluster_yml, image_split_cmd, config_dir)
        print("\n✅ Image split pipeline complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline error: {e}")
        raise
    finally:
        print("\n=== Tear down cluster ===")
        print("$", " ".join(["ray", "down", cluster_yml, "-y"]))
        subprocess.run(["ray", "down", cluster_yml, "-y"], cwd=config_dir)


def run_local_pipeline():
    """Run the pipeline locally with Ray."""
    print("Detected local paths - using local Ray execution...")
    
    # Import ray and initialize
    import ray
    ray.init()
    
    try:
        print("Using Ray-distributed version for better performance...")
        main_with_ray(
            xml_input=XML_INPUT,
            xml_output=XML_OUTPUT,
            n5_output=N5_OUTPUT,
            target_image_size_string=TARGET_IMAGE_SIZE_STRING,
            target_overlap_string=TARGET_OVERLAP_STRING,
            fake_interest_points=FAKE_INTEREST_POINTS,
            fip_density=FIP_DENSITY,
            fip_min_num_points=FIP_MIN_NUM_POINTS,
            fip_max_num_points=FIP_MAX_NUM_POINTS,
            fip_exclusion_radius=FIP_EXCLUSION_RADIUS,
            assign_illuminations=ASSIGN_ILLUMINATIONS,
            disable_optimization=DISABLE_OPTIMIZATION,
            fip_error=FIP_ERROR
        )
    finally:
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()


def run_image_split_pipeline():
    """
    Run the image split pipeline with automatic execution method selection.
    
    This function automatically detects whether input/output paths are S3 URLs
    or local paths and chooses the appropriate execution method.
    """
    # Check if any of the paths are S3 URLs
    if is_s3_path(XML_INPUT) or is_s3_path(XML_OUTPUT) or is_s3_path(N5_OUTPUT):
        run_aws_cluster_pipeline()
    else:
        run_local_pipeline()


if __name__ == "__main__":
    print("Running image split pipeline...")
    print(f"XML Input: {XML_INPUT}")
    print(f"XML Output: {XML_OUTPUT}")
    print(f"N5 Output: {N5_OUTPUT}")
    print()
    
    run_image_split_pipeline()
    print("Image split pipeline completed.")