import subprocess
import json

# To Run - 
# python3 /workspace/Test/input_preparation_test.py

def run_glue_script(bucket_name, xml_file):
    command = [
        'python3', '/workspace/GlueJobs/input_preparation_local.py',
        '--JOB_NAME', 'test_job',
        '--BUCKET_NAME', bucket_name,
        '--XML_FILENAME', xml_file
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    return result.stdout, result.stderr

def main():
    # TO DO - 
    # test with content in PointSpreadFunctions, BoundingBoxes, StitchingResults, IntensityAdjustments attributes
    test_cases = [
        {"bucket_name": "rhapso-example-data-zarr", "xml_file": "missing_sequence_desc_fail.xml", "expected": "fail"},
        {"bucket_name": "rhapso-example-data-zarr", "xml_file": "missing_view_setups_fail.xml", "expected": "fail"},
        {"bucket_name": "rhapso-example-data-zarr", "xml_file": "missing_time_points_fail.xml", "expected": "fail"},
        {"bucket_name": "rhapso-example-data-zarr", "xml_file": "missing_view_registrations_fail.xml", "expected": "fail"},
        {"bucket_name": "rhapso-example-data-zarr", "xml_file": "missing_point_spread_functions_pass.xml", "expected": "pass"},
        {"bucket_name": "rhapso-example-data-zarr", "xml_file": "zarr_all_column_pass.xml", "expected": "pass"},
        {"bucket_name": "rhapso-example-data-zarr", "xml_file": "missing_missing_views_pass.xml", "expected": "pass"},
        {"bucket_name": "rhapso-example-data-zarr", "xml_file": "missing_bounding_boxes_pass.xml", "expected": "pass"},
        {"bucket_name": "rhapso-example-data-zarr", "xml_file": "missing_image_loader_pass.xml", "expected": "pass"},
        {"bucket_name": "rhapso-example-data-zarr", "xml_file": "missing_stitching_results_pass.xml", "expected": "pass"},
        {"bucket_name": "rhapso-example-data-zarr", "xml_file": "missing_intensity_adjustments_pass.xml", "expected": "pass"}
        # {"bucket_name": "rhapso-example-data-zarr", "xml_file": "tiff_all_column_pass.xml", "expected": "pass"}
    ]

    results = {}
    for case in test_cases:
        stdout, stderr = run_glue_script(case['bucket_name'], case['xml_file'])
        test_result = "pass" if "Exception" not in stderr and "Error" not in stderr else "fail"
        results[f"{case['bucket_name']}/{case['xml_file']}"] = {
            'stdout': stdout,
            'stderr': stderr,
            'result': test_result,
            'expected': case['expected'],
            'test_passed': test_result == case['expected']
        }

    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
