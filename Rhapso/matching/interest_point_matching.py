import xml.etree.ElementTree as ET
import cv2
import numpy as np

def matchViews(stuff):
    print("matching views")
    return("apple")

def buildLabelMap(xml_root):
    """
    Build a global label map for all views using the provided XML root element.

    Parameters:
    xml_root (Element): Root element of the XML tree containing view interest points.

    Returns:
    dict: A dictionary mapping ViewId objects to dictionaries of label weights.
    """
    print("building label map")
    
    label_map_global = {}
    label_weights = {}

    # Parse the XML to get view IDs and interest points
    view_setups = xml_root.findall(".//ViewSetup")
    timepoints = xml_root.findall(".//Timepoint")
    
    if not view_setups:
        print("No ViewSetup elements found in the XML.")
    if not timepoints:
        print("No Timepoint elements found in the XML.")
    
    for view_setup in view_setups:
        setup = int(view_setup.find("id").text)
        label = view_setup.find("name").text

        for timepoint in timepoints:
            timepoint_id = int(timepoint.find("id").text)
            view_id = (timepoint_id, setup)

            if label not in label_weights:
                label_weights[label] = 1.0  # Assign a default weight of 1.0 for each label

            if view_id not in label_map_global:
                label_map_global[view_id] = {}

            label_map_global[view_id][label] = label_weights[label]

    return label_map_global, label_weights


'''
Methods
'''

def parse_xml(xml_path):
    # Parse the XML file to extract view IDs, timepoints, and setups
    tree = ET.parse(xml_path)
    root = tree.getroot()
    views = []
    for view in root.findall('.//View'):
        timepoint = int(view.get('timepoint'))
        setup = int(view.get('setup'))
        views.append((timepoint, setup))
    return views

def load_interest_points(views):
    # Load interest points from the dataset based on view IDs
    interest_points = {}
    for view in views:
        timepoint, setup = view
        # Load interest points for the given timepoint and setup
        # This is a placeholder, replace with actual loading logic
        interest_points[view] = np.random.rand(100, 2)  # Example: 100 random points
    return interest_points

def filter_interest_points(interest_points):
    # Filter interest points to keep only those that overlap with other views
    filtered_points = {}
    for view, points in interest_points.items():
        # Placeholder logic for filtering
        filtered_points[view] = points[points[:, 0] > 0.5]  # Example: filter points with x > 0.5
    return filtered_points

def group_views(views):
    # Group views based on specified criteria
    grouped_views = {}
    for view in views:
        timepoint, setup = view
        if timepoint not in grouped_views:
            grouped_views[timepoint] = []
        grouped_views[timepoint].append(setup)
    return grouped_views

def setup_pairwise_matching(grouped_views):
    # Set up pairwise matching tasks
    pairs = []
    for timepoint, setups in grouped_views.items():
        for i in range(len(setups)):
            for j in range(i + 1, len(setups)):
                pairs.append(((timepoint, setups[i]), (timepoint, setups[j])))
    return pairs

def ransac_parameters():
    # Set up RANSAC parameters
    params = {
        'max_iterations': 10000,
        'max_error': 5.0,
        'min_inlier_ratio': 0.1,
        'min_inlier_factor': 3.0
    }
    return params

def create_matcher(method, params):
    # Create an instance of the matcher based on the specified parameters
    if method == 'FAST_ROTATION':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif method == 'FAST_TRANSLATION':
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        raise ValueError("Unsupported method")
    return matcher

def compute_pairwise_match(matcher, points1, points2):
    # Compute the pairwise match between two sets of interest points
    matches = matcher.match(points1, points2)
    return matches

def add_correspondences(matches, interest_points):
    # Add correspondences between interest points to the dataset
    correspondences = []
    for match in matches:
        correspondences.append((interest_points[match.queryIdx], interest_points[match.trainIdx]))
    return correspondences

def save_results(correspondences, output_path):
    # Save the resulting correspondences and matched interest points to the output file
    with open(output_path, 'w') as f:
        for corr in correspondences:
            f.write(f"{corr[0]} {corr[1]}\n")

# Example usage
# xml_path = "/path/to/dataset.xml"
# views = parse_xml(xml_path)
# interest_points = load_interest_points(views)
# filtered_points = filter_interest_points(interest_points)
# grouped_views = group_views(views)
# pairs = setup_pairwise_matching(grouped_views)
# params = ransac_parameters()
# matcher = create_matcher('FAST_ROTATION', params)

# for pair in pairs:
#     points1 = filtered_points[pair[0]]
#     points2 = filtered_points[pair[1]]
#     matches = compute_pairwise_match(matcher, points1, points2)
#     correspondences = add_correspondences(matches, interest_points)
#     save_results(correspondences, "output.txt")

# import sys
# from awsglue.transforms import *
# from awsglue.utils import getResolvedOptions
# from pyspark.context import SparkContext
# from awsglue.context import GlueContext
# from awsglue.job import Job
# from awsglue.utils import getResolvedOptions
# from awsglue.dynamicframe import DynamicFrame
# from pyspark.sql import SparkSession
# from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame
# from Rhapso.detection.view_transform_models import ViewTransformModels
# from Rhapso.detection.overlap_detection import OverlapDetection
# from Rhapso.data_prep.load_image_data import LoadImageData
# from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian
# from Rhapso.data_prep.serialize_image_chunks import SerializeImageChunks
# from Rhapso.data_prep.deserialize_image_chunks import DeserializeImageChunks
# from Rhapso.data_prep.glue_crawler import GlueCrawler
# import boto3
# import base64
# import numpy as np
# import io
def build_label_map(data, view_ids, label_weights):
    """
    Build a global label map for all views using the provided data and view IDs.

    Parameters:
    data (SpimData2): The SpimData2 object containing view interest points.
    view_ids (list of ViewId): List of ViewId objects representing the views.
    label_weights (dict): Dictionary mapping labels to their weights.

    Returns:
    dict: A dictionary mapping ViewId objects to dictionaries of label weights.
    """
    label_map_global = {}

    for view_id in view_ids:
        view_interest_points = data.get_view_interest_points().get_view_interest_point_lists(view_id)

        # Ensure the label exists for all views that should be processed
        for label in label_weights.keys():
            if view_interest_points.get_interest_point_list(label) is None:
                print(f"Error, label '{label}' does not exist for ViewId {view_id}")
                exit(1)

        # Needs to be ViewId, not ViewDescription, then it's serializable
        label_map_global[view_id] = label_weights

    return label_map_global
# # Spark ETL testing pipeline - this script runs in AWS Glue

# s3 = boto3.client('s3')

# ## @params: [JOB_NAME]
# args = getResolvedOptions(sys.argv, ['JOB_NAME'])
# sc = SparkContext.getOrCreate() 
# glueContext = GlueContext(sc)
# spark = SparkSession.builder.appName("Interest Point Detection").getOrCreate()
# job = Job(glueContext)
# job.init(args['JOB_NAME'], args)

# # DOWNSAMPLING
# dsxy = 4
# dsz = 2
# strategy = 'spark-etl'

# # SPARK ETL PARAMS
# parquet_bucket_path = 's3://interest-point-detection/ipd-staging/'
# crawler_name = "InterestPointDetectionCrawler"
# crawler_s3_path = "s3://interest-point-detection/ipd-staging/"
# crawler_database_name = "interest_point_detection"
# crawler_iam_role = "arn:aws:iam::443370675126:role/rhapso-s3"
# glue_database = 'interest_point_detection'
# glue_table_name = 'ipd_staging'

# # FILE TYPE - PICK ONE
# file_type = 'tiff'
# xml_bucket_name = "rhapso-tif-sample"
# image_bucket_name = "tiff-sample"
# prefix = 's3://rhapso-tif-sample/IP_TIFF_XML/'
# xml_file_path = "IP_TIFF_XML/dataset.xml"

# # file_type_zarr = 'zarr'
# # xml_bucket_name = "rhapso-zar-sample"
# # image_bucket_name = "aind-open-data"
# # prefix = 's3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr'
# # xml_file_path = "dataset.xml"

# def fetch_from_s3(s3, bucket_name, input_file):
#     response = s3.get_object(Bucket=bucket_name, Key=input_file)
#     return response['Body'].read().decode('utf-8')

# # INTEREST POINT DETECTION
# # --------------------------

# xml_file = fetch_from_s3(s3, xml_bucket_name, xml_file_path) 

# # Load XML data into dataframes
# processor = XMLToDataFrame(xml_file)
# dataframes = processor.run()
# print("XML loaded")

# # Create view transform matrices 
# create_models = ViewTransformModels(dataframes)
# view_transform_matrices = create_models.run()
# print("Transforms Models have been created")

# # Use view transform matrices to find areas of overlap
# overlap_detection = OverlapDetection(view_transform_matrices, dataframes, dsxy, dsz, prefix, file_type)
# overlapping_area = overlap_detection.run()
# print("Overlap Detection is done")

# # Load images
# images_loader = LoadImageData(dataframes, overlapping_area, dsxy, dsz, prefix, file_type)
# all_image_data = images_loader.run()
# print("Image data has loaded")

# # Flatten and serialize images to parquet
# serialize_image_chunks = SerializeImageChunks(all_image_data, parquet_bucket_path)
# serialized_images_dyf = serialize_image_chunks.run()
# print("Serialized image data")

# # Create and start crawler
# glue_crawler = GlueCrawler(crawler_name, crawler_s3_path, crawler_database_name, crawler_iam_role)
# glue_crawler.run()
# print("Glue crawler created and started")

# # Create dynamic frame using crawler schema
# image_data_dyf = glueContext.create_dynamic_frame.from_catalog(
#     database = glue_database,
#     table_name = glue_table_name,
#     transformation_ctx = "dynamic_frame"
# )
# print("Dynamic frame loaded")

# # Detect interest points using DoG algorithm - custom transform
# difference_of_gaussian = DifferenceOfGaussian()
# deserialize_image_chunks = DeserializeImageChunks()
# def interest_point_detection(record):
#     image_chunk = deserialize_image_chunks.run(record)
#     interest_points = difference_of_gaussian.run(image_chunk)
#     interest_points_as_strings = [str(point) for point in interest_points]
#     return {'interest_points': interest_points_as_strings}
# mapped_results_dyf = image_data_dyf.map(interest_point_detection, transformation_ctx="map_interest_points")
# print("Interest point detection done")

# # View results
# result_df = mapped_results_dyf.toDF()
# result_df.show()

# # filtering_and_optimizing = FilteringAndOptimizing(interest_points)
# # filtering_and_optimizing.run()

# # advanced_refinement = AdvancedRefinement()
# # advanced_refinement.run()

# # save_interest_points = SaveInterestPoints()
# # save_interest_points.run()

# # INTEREST POINT MATCHING
# # --------------------------

# # SOLVE 
# # --------------------------

# # FUSION
# # --------------------------

# job.commit()