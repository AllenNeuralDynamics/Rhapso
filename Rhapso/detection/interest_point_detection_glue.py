from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso.data_prep.load_image_data import LoadImageData
from Rhapso.data_prep.glue_crawler import GlueCrawler
from Rhapso.data_prep.serialize_image_chunks import SerializeImageChunks
from Rhapso.data_prep.deserialize_image_chunks import DeserializeImageChunks
from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian
from Rhapso.detection.advanced_refinement import AdvancedRefinement
from Rhapso.detection.save_interest_points import SaveInterestPoints
import boto3
import ast

# This class implements the interest point detection pipeline

class InterestPointDetectionGlue:
    def __init__(self, strategy, dsxy, dsz, min_intensity, max_intensity, sigma, threshold, file_type, file_source,
            xml_file_path, xml_bucket_name, image_file_path, image_bucket_name, output_file_path, output_bucket_name,
            parquet_bucket_path, crawler_name, crawler_s3_path, crawler_database_name, crawler_iam_role, glue_database,
            glue_table_name, glue_context):
        
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
        self.parquet_bucket_path = parquet_bucket_path
        self.crawler_name = crawler_name
        self.crawler_s3_path = crawler_s3_path
        self.crawler_database_name = crawler_database_name
        self.crawler_iam_role = crawler_iam_role
        self.glue_database = glue_database
        self.glue_table_name = glue_table_name
        self.glue_context = glue_context
        self.s3 = boto3.client('s3')

    def detection(self):
        # data input source

        def fetch_from_s3(bucket_name, input_file):
            response = self.s3.get_object(Bucket=bucket_name, Key=input_file)
            return response['Body'].read().decode('utf-8')

        def fetch_local_xml(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()

        # INTEREST POINT DETECTION
        # --------------------------

        # Fetch xml data
        if self.file_source == 's3':
            xml_file = fetch_from_s3(self.xml_bucket_name, self.xml_file_path) 
        elif self.file_source == 'local':
            xml_file = fetch_local_xml(self.xml_file_path)

        # Load XML data into dataframes         
        processor = XMLToDataFrame(xml_file)
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

        # Flatten and serialize images to parquet
        serialize_image_chunks = SerializeImageChunks(all_image_data, self.parquet_bucket_path)
        serialize_image_chunks.run()
        print("Serialized image data")

        # Create and start crawler
        glue_crawler = GlueCrawler(self.crawler_name, self.crawler_s3_path, self.crawler_database_name, self.crawler_iam_role)
        glue_crawler.run()
        print("Glue crawler created and started")

        # Create dynamic frame using crawler schema
        image_data_dyf = self.glue_context.create_dynamic_frame.from_catalog(
            database = self.glue_database,
            table_name = self.glue_table_name,
            transformation_ctx = "dynamic_frame"
        )
        print("Dynamic frame loaded")

        # Detect interest points using DoG algorithm
        difference_of_gaussian = DifferenceOfGaussian(self.min_intensity, self.max_intensity, self.sigma, self.threshold)

        # Detect interest points using DoG algorithm - custom transform
        difference_of_gaussian = DifferenceOfGaussian(self.min_intensity, self.max_intensity, self.sigma, self.threshold)
        deserialize_image_chunks = DeserializeImageChunks()
        def interest_point_detection(record):
            try:
                view_id, interval, image_chunk = deserialize_image_chunks.run(record)     
                dog_results = difference_of_gaussian.run(image_chunk, self.dsxy, self.dsz)
                interest_points = dog_results['interest_points']
                intensities = dog_results['intensities']
                interest_points_as_strings = [str(point) for point in interest_points]
                results_dict = {
                    'view_id': str(view_id),
                    'interval_key': str(interval),
                    'interest_points': interest_points_as_strings,
                    'intensities': intensities.tolist() 
                }
                return results_dict
            except Exception as e:
                print("Error processing record:", str(e))
                return {}
        mapped_results_dyf = image_data_dyf.map(interest_point_detection, transformation_ctx="map_interest_points")
        print("Difference of gaussian is done")

        # Format results out of dynamic frame for advanced refinement
        result_df = mapped_results_dyf.toDF()
        interest_points_list = []
        for row in result_df.collect():
            view_id = row['view_id']
            interval_key = row['interval_key']
            interest_points = [ast.literal_eval(point) for point in row['interest_points']]
            intensities = row['intensities']
            interest_points_list.append({
                'view_id': view_id,
                'interval_key': interval_key,
                'interest_points': interest_points,
                'intensities': intensities
            })
        print("Results formatted and ready for advanced refinement")

        # Integrate final peaks into kd tree for refinement
        advanced_refinement = AdvancedRefinement(interest_points_list)
        consolidated_data = advanced_refinement.run()
        print("Advanced refinement is complete.")

        # Save interest points to N5 and metadata to XML
        save_interest_points = SaveInterestPoints(dataframes, consolidated_data, self.xml_file_path, self.xml_bucket_name, 
                                                self.output_bucket_name, self.output_file_path, self.dsxy, self.dsz, self.min_intensity, 
                                                self.max_intensity, self.sigma, self.threshold, self.file_source)
        save_interest_points.run()
        print("Interest points saved")

        print("Interest point detection is done")
    
    def run(self):
        self.detection()