"""
Run as module: `python3 -m Rhapso.pipelines.image_split.image_split_pipeline`
"""
print("Begin image split pipeline.")

# Import the main function from split_datasets
from Rhapso.image_split.split_datasets import main

# Configuration Variables

# Local Input/Output Paths
XML_INPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/bigstitcher_affine.xml"
XML_OUTPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/bigstitcher_affine_split.xml"
N5_OUTPUT = "/mnt/c/Users/marti/Documents/allen/rhapso-e2e-testing/split_verification/ip_affine_alignment/interestpoints.n5"

# XML_INPUT = "/Users/seanfite/Desktop/interest_point_detection/rhapso-solver-affine.xml"
# XML_OUTPUT = "/Users/seanfite/Desktop/interest_point_detection/rhapso-solver-split-affine.xml"
# N5_OUTPUT = "/Users/seanfite/Desktop/interest_point_detection/interestpoints.n5"

# AWS S3 Input/Output Paths
# XML_INPUT = "s3://martin-test-bucket/split_images_output/bigstitcher_affine.xml"
# XML_OUTPUT = "s3://martin-test-bucket/split_images_output/output.xml"
# N5_OUTPUT = "s3://martin-test-bucket/split_images_output/interestpoints.n5"

# XML_INPUT = "s3://rhapso-matching-test/rhapso-solver-affine.xml"
# XML_OUTPUT = "s3://rhapso-matching-test/rhapso-solver-split-affine"
# N5_OUTPUT = "s3://rhapso-matching-test/interestpoints.n5"

# Image Splitting Parameters
TARGET_IMAGE_SIZE_STRING = "7000,7000,4000"
TARGET_OVERLAP_STRING = "128,128,128"

# Fake Interest Points Configuration
FAKE_INTEREST_POINTS = True
FIP_EXCLUSION_RADIUS = 200
ASSIGN_ILLUMINATIONS = True

# Default Values
DISABLE_OPTIMIZATION = False
FIP_DENSITY = 100.0
FIP_MIN_NUM_POINTS = 20
FIP_MAX_NUM_POINTS = 500
FIP_ERROR = 0.5

if __name__ == "__main__":
    print("Running image split pipeline...")
    main(
        xml_input=XML_INPUT,
        xml_output=XML_OUTPUT,
        n5_output=N5_OUTPUT,
        target_image_size_string=TARGET_IMAGE_SIZE_STRING,
        target_overlap_string=TARGET_OVERLAP_STRING,
        fake_interest_points=FAKE_INTEREST_POINTS,
        fip_exclusion_radius=FIP_EXCLUSION_RADIUS,
        assign_illuminations=ASSIGN_ILLUMINATIONS,
        disable_optimization=DISABLE_OPTIMIZATION,
        fip_density=FIP_DENSITY,
        fip_min_num_points=FIP_MIN_NUM_POINTS,
        fip_max_num_points=FIP_MAX_NUM_POINTS,
        fip_error=FIP_ERROR
    )
    print("Image split pipeline completed.")
else:
    print("Image split pipeline module imported.")