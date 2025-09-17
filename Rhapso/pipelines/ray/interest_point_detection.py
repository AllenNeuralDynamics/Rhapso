from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso.detection.metadata_builder import MetadataBuilder
from Rhapso.detection.image_reader import ImageReader
from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian
from Rhapso.detection.advanced_refinement import AdvancedRefinement
from Rhapso.detection.points_validation import PointsValidation
from Rhapso.detection.save_interest_points import SaveInterestPoints
import boto3
import ray

# This class implements the interest point detection pipeline

class InterestPointDetection:
    def __init__(self, dsxy, dsz, min_intensity, max_intensity, sigma, threshold, file_type, xml_file_path, 
                 image_file_prefix, xml_output_file_path, n5_output_file_prefix, combine_distance, chunks_per_bound, run_type, 
                 max_spots, median_filter):
        self.dsxy = dsxy
        self.dsz = dsz
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.sigma = sigma
        self.threshold = threshold
        self.file_type = file_type
        self.xml_file_path = xml_file_path
        self.image_file_prefix = image_file_prefix
        self.xml_output_file_path = xml_output_file_path
        self.n5_output_file_prefix = n5_output_file_prefix
        self.combine_distance = combine_distance
        self.chunks_per_bound = chunks_per_bound
        self.run_type = run_type
        self.max_spots = max_spots
        self.median_filter = median_filter

    def detection(self):
        # Get XML file
        if self.xml_file_path.startswith("s3://"):
            no_scheme = self.xml_file_path.replace("s3://", "", 1)
            bucket, key = no_scheme.split("/", 1)
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=bucket, Key=key)
            xml_file = response["Body"].read().decode("utf-8")

        else:  
            with open(self.xml_file_path, "r", encoding="utf-8") as f:
                xml_file = f.read()

        # Load XML data into dataframes         
        processor = XMLToDataFrame(xml_file)
        dataframes = processor.run()
        print("XML loaded")

        # Create view transform matrices 
        create_models = ViewTransformModels(dataframes)
        view_transform_matrices = create_models.run()
        print("Transforms models have been created")

        # Use view transform matrices to find areas of overlap
        overlap_detection = OverlapDetection(view_transform_matrices, dataframes, self.dsxy, self.dsz, self.image_file_prefix, self.file_type)
        overlapping_area, new_dsxy, new_dsz, level, max_interval_size, mip_map_downsample = overlap_detection.run()
        print("Overlap detection is done")

        # Implement image chunking strategy as list of metadata 
        metadata_loader = MetadataBuilder(dataframes, overlapping_area, self.image_file_prefix, self.file_type, new_dsxy, new_dsz, 
                                          self.chunks_per_bound, self.sigma, self.run_type, level)
        image_chunk_metadata = metadata_loader.run()
        print("Metadata has loaded")

        # Use Ray to distribute peak detection to image chunking metadata 
        @ray.remote
        def process_peak_detection_task(chunk_metadata, new_dsxy, new_dsz, min_intensity, max_intensity, sigma, threshold,
                                        median_filter, mip_map_downsample):
            try:
                difference_of_gaussian = DifferenceOfGaussian(min_intensity, max_intensity, sigma, threshold, median_filter, mip_map_downsample)
                image_fetcher = ImageReader(self.file_type)
                view_id, interval, image_chunk, offset, lb = image_fetcher.run(chunk_metadata, new_dsxy, new_dsz)
                interest_points = difference_of_gaussian.run(image_chunk, offset, lb)

                return {
                    'view_id': view_id,
                    'interval_key': interval,
                    'interest_points': interest_points['interest_points'],
                    'intensities': interest_points['intensities']
                }
            except Exception as e:
                return {'error': str(e), 'view_id': chunk_metadata.get('view_id', 'unknown')}

        # Submit tasks to Ray
        futures = [process_peak_detection_task.remote(chunk_metadata, new_dsxy, new_dsz, self.min_intensity, self.max_intensity, 
                                                      self.sigma, self.threshold, self.median_filter, mip_map_downsample)
            for chunk_metadata in image_chunk_metadata
        ]

        # Gather and process results
        results = ray.get(futures)
        final_peaks = [r for r in results if 'error' not in r]
        print("Peak detection is done")

        # Consolidate points and filter overlap duplicates using kd tree
        advanced_refinement = AdvancedRefinement(final_peaks, self.combine_distance, dataframes, overlapping_area, max_interval_size, self.max_spots)
        consolidated_data = advanced_refinement.run()
        print("Advanced refinement is done")

        # Print points metrics / validation tools
        points_validation = PointsValidation(consolidated_data)
        points_validation.run()
        print("Points metrics printed")

        # Save final interest points
        save_interest_points = SaveInterestPoints(dataframes, consolidated_data, self.xml_file_path, self.xml_output_file_path, self.n5_output_file_prefix, 
                                                  self.dsxy, self.dsz, self.min_intensity, self.max_intensity, self.sigma, self.threshold)
        save_interest_points.run()
        print("Interest points saved")
    
    def run(self):
        self.detection()



# DEBUG - to step through DOG
# final_peaks = []
# for chunk_metadata in image_chunk_metadata:
#     difference_of_gaussian = DifferenceOfGaussian(
#         self.min_intensity, self.max_intensity, self.sigma, self.threshold, self.median_filter, mip_map_downsample
#     )
#     image_fetcher = ImageReader(self.file_type)

#     view_id, interval, image_chunk, offset, lb = image_fetcher.run(chunk_metadata, new_dsxy, new_dsz)
#     interest_points = difference_of_gaussian.run(image_chunk, offset, lb)

#     final_peaks.append({
#         'view_id': view_id,
#         'interval_key': interval,
#         'interest_points': interest_points['interest_points'],
#         'intensities': interest_points['intensities'],
#     })
