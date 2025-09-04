"""
Ray-distributed image split pipeline for Rhapso.

This pipeline uses Ray to parallelize the N5 file creation step for fake interest points,
significantly improving performance when processing large numbers of image tiles.

The pipeline automatically detects if Ray is available and falls back to sequential
processing if not. It also handles Ray initialization and cleanup automatically.

Usage:
    python image_split_pipeline.py

Configuration:
    Edit the configuration variables at the top of this file to customize
    input/output paths and processing parameters.
"""

import os
from Rhapso.image_split.split_datasets import main, main_with_ray

# ============================================================================ 
# Configuration
# ============================================================================

# Input/Output Paths
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

# Ray Configuration
USE_RAY_DISTRIBUTION = True  # Set to False to use original sequential version


def run_image_split_pipeline():
    """
    Run the image split pipeline with optional Ray distribution.
    
    This function automatically chooses between Ray-distributed and sequential
    processing based on the USE_RAY_DISTRIBUTION configuration flag.
    """
    if USE_RAY_DISTRIBUTION:
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
    else:
        print("Using original sequential version...")
        main(
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


if __name__ == "__main__":
    print("Running image split pipeline...")
    run_image_split_pipeline()
    print("Image split pipeline completed.")