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
from pathlib import Path

# ============================================================================ 
# Configuration
# ============================================================================

# AWS S3 Input/Output Paths
XML_INPUT = "s3://martin-test-bucket/split-images-test/bigstitcher_affine.xml"
XML_OUTPUT = "s3://martin-test-bucket/split-images-test/bigstitcher_affine_split.xml"
N5_OUTPUT = "s3://martin-test-bucket/split-images-test/interestpoints.n5"

# Local Input/Output Paths (commented out)
# XML_INPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/bigstitcher_affine.xml"
# XML_OUTPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/bigstitcher_affine_split.xml"
# N5_OUTPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/interestpoints.n5"

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
    
    # Get the directory of this script
    script_dir = Path(__file__).resolve().parent
    
    # Run the AWS version
    aws_script = script_dir / "aws" / "image_split_pipeline.py"
    subprocess.run([sys.executable, str(aws_script)], check=True)


def run_local_pipeline():
    """Run the pipeline locally with Ray."""
    print("Detected local paths - using local Ray execution...")
    
    # Get the directory of this script
    script_dir = Path(__file__).resolve().parent
    
    # Run the local version
    local_script = script_dir / "local" / "image_split_pipeline.py"
    subprocess.run([sys.executable, str(local_script)], check=True)


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