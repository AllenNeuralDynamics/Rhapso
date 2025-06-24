from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso.data_prep.load_image_data import LoadImageData
from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian
from Rhapso.detection.advanced_refinement import AdvancedRefinement
from Rhapso.detection.save_interest_points import SaveInterestPoints
import boto3
from dask import delayed
from dask import compute

# This class implements the interest point detection pipeline

class InterestPointDetection:
    def __init__(self, strategy, dsxy, dsz, min_intensity, max_intensity, sigma, threshold, file_type, file_source,
            xml_file_path, xml_bucket_name, image_file_path, image_bucket_name, output_file_path, output_bucket_name, metrics_output_path,
            key):
        
        self.strategy = strategy
        self.dsxy = dsxy
        self.dsz = dsz
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.sigma = sigma
        self.threshold = threshold
        self.file_type = file_type
        self.file_source = file_source
        self.xml_file_path = xml_file_path
        self.xml_bucket_name = xml_bucket_name
        self.image_file_path = image_file_path
        self.image_bucket_name = image_bucket_name
        self.output_file_path = output_file_path
        self.output_bucket_name = output_bucket_name
        self.metrics_output_path = metrics_output_path
        self.key = key

    def detection(self):
        # data input source
        s3 = boto3.client("s3")

        # INTEREST POINT DETECTION
        # --------------------------
        def fetch_from_s3(s3, bucket_name, input_file):
            response = s3.get_object(Bucket=bucket_name, Key=input_file)
            return response["Body"].read().decode("utf-8")

        def fetch_local_xml(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()

        # Fetch xml data
        if self.file_source == "s3":
            xml_file = fetch_from_s3(s3, self.xml_bucket_name, self.xml_file_path)
        elif self.file_source == "local":
            xml_file = fetch_local_xml(self.xml_file_path)

        # Load XML data into dataframes
        processor = XMLToDataFrame(xml_file, self.key)
        dataframes = processor.run()
        print("XML loaded")

        # Create view transform matrices
        create_models = ViewTransformModels(dataframes)
        view_transform_matrices = create_models.run()
        print("Transforms models have been created")

        # Use view transform matrices to find areas of overlap
        overlap_detection = OverlapDetection(view_transform_matrices, dataframes, self.dsxy, self.dsz, self.image_file_path, self.file_type)
        overlapping_area = overlap_detection.run()
        print("Overlap detection is done")

        # Load images 
        images_loader = LoadImageData(dataframes, overlapping_area, self.dsxy, self.dsz, self.image_file_path, self.file_type)
        all_image_data = images_loader.run()
        print("Image data loaded")

        # Detect interest points using DoG algorithm
        difference_of_gaussian = DifferenceOfGaussian(self.min_intensity, self.max_intensity, self.sigma, self.threshold)

        # DASK MAP VERSION (DO NOT DISTRIBUTE)
        final_peaks = []
        delayed_results = []
        delayed_keys = {}
        for image_data in all_image_data:
            view_id = image_data['view_id']
            interval_key = image_data['interval_key']
            image_chunk = image_data['image_chunk']
            dog_result = delayed(difference_of_gaussian.run)(image_chunk, self.dsxy, self.dsz)
            delayed_results.append(dog_result)
            delayed_keys[dog_result] = (view_id, interval_key)
        computed_results = compute(*delayed_results)
        for result, task in zip(computed_results, delayed_results):
            view_id, interval_key = delayed_keys[task]
            final_peaks.append(
                {
                    "view_id": view_id,
                    "interval_key": interval_key,
                    "interest_points": result["interest_points"],
                    "intensities": result["intensities"],
                }
            )
        print("Difference of gaussian is done")

        # Integrate final peaks into kd tree for refinement
        advanced_refinement = AdvancedRefinement(final_peaks)
        consolidated_data = advanced_refinement.run()
        print("Advanced refinement is done")

        # Save interest points to N5 and metadata to XML
        save_interest_points = SaveInterestPoints(dataframes, consolidated_data, self.xml_file_path, self.xml_bucket_name, 
                                                self.output_bucket_name, self.output_file_path, self.dsxy, self.dsz, self.min_intensity, 
                                                self.max_intensity, self.sigma, self.threshold, self.file_source)
        save_interest_points.run()
        print("Interest points saved")

        print("Interest point detection is done")
    
    def run(self):
        self.detection()