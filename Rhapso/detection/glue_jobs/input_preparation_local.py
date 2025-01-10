import sys
import os
import boto3
import numpy as np
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.sql.utils import AnalysisException
from pyspark.sql.functions import explode, col
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession

# To Run: 
# docker start ee0480b5bfe9
# docker attach ee0480b5bfe9
# python3 /workspace/GlueJobs/input_preparation_local.py --JOB_NAME input_preparation_local --BUCKET_NAME rhapso-example-data-zarr --XML_FILENAME dataset.xml

# This version is designed to run locally

class Initialize:
    def __init__(self, args):
        self.job_name = args['JOB_NAME']
        self.bucket_name = args['BUCKET_NAME']
        self.xml_filename = args['XML_FILENAME']
        self.s3_input_path = f"s3://{self.bucket_name}/{self.xml_filename}"
        self.output_path = f"s3://{self.bucket_name}/parallel-processing/"
        self.base_path_uri = self.base_path_URI(self.s3_input_path)
        self.df = None
        self.setup_contexts()
    
    def base_path_URI(self, file_key: str) -> str:
        directory_path = os.path.dirname(file_key)
        return f"s3://{self.bucket_name}/{directory_path}" if directory_path else f"s3://{self.bucket_name}/"
    
    def setup_contexts(self):
        if '--JOB_NAME' not in sys.argv: sys.argv.extend(['--JOB_NAME', self.job_name])
        
        self.args = getResolvedOptions(sys.argv, ['JOB_NAME'])
        self.spark = SparkSession.builder \
            .appName(self.args['JOB_NAME']) \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        self.glueContext = GlueContext(self.spark.sparkContext)
        self.job = Job(self.glueContext)
        
        if 'AWS_EXECUTION_ENV' in os.environ:
            self.job.init(self.args['JOB_NAME'], self.args)
        else:
            print("Running locally, skipping job.init")

    def load_data(self):
        dyf = self.glueContext.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={"paths": [self.s3_input_path]},
            format="xml",
            format_options={"rowTag": "SpimData"}
        )
        self.df = dyf.toDF()

        return self.df
    
class SequenceDescription:
    def __init__(self, dataframe, bucket_name):
        self.df = dataframe
        self.bucket_name = bucket_name
     
    def load_view_setups(self):
        try:
            exploded_df = self.df.select(explode("SequenceDescription.ViewSetups.ViewSetup").alias("view_setup"))  
            
            if exploded_df.rdd.isEmpty():
                raise ValueError("No view setups found in DataFrame.")  
            
            view_setups_dict = {
                int(row.view_setup.id): row.view_setup.asDict() 
                for row in exploded_df.collect()
                if row.view_setup and row.view_setup.id is not None
            }

            if not view_setups_dict:
                raise ValueError("View setups dictionary is empty after processing.")

            return view_setups_dict
        
        except AnalysisException as e:
            raise ValueError(f"Error processing DataFrame: {e}")
    
    def load_time_points(self):
        try:
            timepoints_df = self.df.select("SequenceDescription.Timepoints.*")
            if timepoints_df.rdd.isEmpty():
                raise ValueError("Timepoints data is missing in the DataFrame.")
            
            result = {}
            timepoints_type = timepoints_df.first()._type

            if timepoints_type == "pattern":
                pattern = timepoints_df.first().integerpattern
                result['pattern'] = pattern

            elif timepoints_type == "range":
                first = timepoints_df.first().first
                last = timepoints_df.first().last
                result['range'] = {t: {'timepoint': t} for t in range(first, last + 1)}

            elif timepoints_type == "list":
                list_ids = timepoints_df.select(explode("id").alias("id"))
                result['list'] = {row.id: {'timepoint': row.id} for row in list_ids.collect()}

            else:
                raise ValueError(f"Unknown timepoints type: {timepoints_type}")

            return result
        except AnalysisException as e:
            raise ValueError(f"Error accessing timepoints data: {e}")

    def load_missing_views(self):
        if 'MissingViews' in self.df.columns:
            try:
                missing_views_df = self.df.select(explode("SequenceDescription.MissingViews.view").alias("missing_view"))
                views = set(
                    (row.missing_view.timepoint, row.missing_view.setup)
                    for row in missing_views_df.collect()
                    if row.missing_view is not None
                )

                return views
            
            except AnalysisException as e:
                print(f"Error processing DataFrame for missing views: {e}")
                return set()
            
        else:
            return set()
    
    def zarr_image_loader(self, sequence_description):
        try:
            s3_bucket_element = self.df.select("SequenceDescription.ImageLoader.s3bucket").first()
            s3_client = boto3.client('s3') if 's3_client_needed' in os.environ else None

            bucket = None
            folder = None

            zarr_folder_element = self.df.select("SequenceDescription.ImageLoader.zarr").first()

            if zarr_folder_element and hasattr(zarr_folder_element[0], "_VALUE"):
                folder = zarr_folder_element[0]._VALUE
            else:
                folder = str(zarr_folder_element[0]) if zarr_folder_element else ""

            if not folder.startswith("/"):
                folder = "/" + folder
            if not folder.endswith("/"):
                folder += "/"

            if s3_bucket_element and s3_bucket_element[0]:
                bucket = s3_bucket_element[0]
                uri = f"s3://{bucket}{folder}"
            else:
                uri = folder  

            zgroups_df = self.df.select(explode("SequenceDescription.ImageLoader.zgroups.zgroup").alias("zgroup"))
            zgroups = {
                (int(row.zgroup['_timepoint']), int(row.zgroup['_setup'])): row.zgroup['path']
                for row in zgroups_df.collect()
                if row.zgroup['path'] is not None
            }

            keyValueReader = {
                'multi_scale_path': uri,
                's3_client': s3_client,
                'bucket_name': bucket,
                's3_mode': bool(bucket) 
            }

            return {
                'z_groups': zgroups,
                'seq': sequence_description,
                'zarr_key_value_reader_builder': keyValueReader
            }
        
        except Exception as e:
            print(f"❌ Error in Zarr Image Loader: {e}")
            return {}
    
    def tiff_image_loader(self, sequence_description):
        try:
            zgrouped = self.df.select("SequenceDescription.ImageLoader._zgrouped").first()
            s3_bucket_element = self.df.select("SequenceDescription.ImageLoader.s3bucket").first()
            bucket = s3_bucket_element[0] if s3_bucket_element and s3_bucket_element[0] else None
            s3_client = boto3.client('s3') if 's3_client_needed' in os.environ else None

            file_map_df = self.df.select(explode("SequenceDescription.ImageLoader.files.fileMapping").alias("fileMapping"))
            file_map = {}
            for row in file_map_df.collect():
                vs = int(row.fileMapping['_vs'])
                tp = int(row.fileMapping['_tp'])
                series = int(row.fileMapping['_series'])
                channel = int(row.fileMapping['_channel'])
                file_path = row.fileMapping['file']

                if bucket and not file_path.startswith("s3://"):
                    file_path = f"s3://{bucket}/{file_path.lstrip('/')}"

                view_id = (tp, vs)

                file_map[view_id] = {
                    'file_path': file_path,
                    'series': series,
                    'channel': channel
                }

            tiffImageLoader = {
                'file_map': file_map,
                'seq': sequence_description,
                's3_client': s3_client,
                's3_mode': bool(bucket),
                'z_grouped': bool(zgrouped[0]) if zgrouped else False
            }

            return tiffImageLoader

        except AnalysisException as e:
            print(f"❌ Error processing TIFF Image Loader: {e}")
            return {}
        except ValueError as e:
            print(f"❌ Configuration Error: {e}")
            return {}
        except Exception as e:
            print(f"❌ Unexpected error while loading TIFF image loader: {e}")
            return {}
    
    def load_image_loader(self, sequence_description):
        try:
            image_loader_field = next(
                (field for field in self.df.schema["SequenceDescription"].dataType.fields if field.name == "ImageLoader"),
                None
            )
            if image_loader_field is None: return {}

            image_loader_format = self.df.select("SequenceDescription.ImageLoader._format").first()[0]
            if image_loader_format == "bdv.multimg.zarr":
                return self.zarr_image_loader(sequence_description)
            elif image_loader_format == "spimreconstruction.filemap2":
                return self.tiff_image_loader(sequence_description)
            else:
                raise ValueError(f"Unsupported image format: {image_loader_format}")

        except AnalysisException as e:
            print(f"Error processing zgroups: {e}")
            return {}
        except ValueError as e:
            print(f"Configuration Error: {e}")
            return {}
        except Exception as e:
            print(f"Unexpected error while loading image loader: {e}")
            return {}
    
    def run(self):
        sequence_description = {}
        sequence_description['view_setups'] = self.load_view_setups()
        sequence_description['time_points'] = self.load_time_points()
        sequence_description['missing_views'] = self.load_missing_views()
        sequence_description['image_loader'] = self.load_image_loader(sequence_description) 
        return sequence_description

class ViewRegistrations:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def load_view_registrations(self):
        registration_df = self.df.select(
            explode("ViewRegistrations.ViewRegistration").alias("registration")
        )

        if registration_df.rdd.isEmpty():
            raise ValueError("No view registrations found in the DataFrame.")

        registration_processed_df = registration_df.select(
            col("registration._timepoint").alias("timepoint_id"),
            col("registration._setup").alias("setup_id"),
            col("registration.ViewTransform._type").alias("transform_type"),
            col("registration.ViewTransform.Name").alias("transform_name"),
            col("registration.ViewTransform.affine").alias("affine_data")
        )

        def process_transform(transform_type, name, affine_data):
            if transform_type == 'affine' and affine_data:
                try:
                    affine_matrix = list(map(float, affine_data.split()))
                    if len(affine_matrix) == 12:  # Ensuring 3x4 matrix size for affine transforms
                        affine_matrix = [affine_matrix[i:i+4] for i in range(0, len(affine_matrix), 4)]
                        return {'type': 'affine', 'name': name, 'affine': affine_matrix}
                except ValueError:
                    return {'type': 'error', 'info': 'Invalid affine data'}
            elif transform_type == 'generic':
                return {'type': 'generic', 'class': name}
            else:
                return {'type': 'unknown', 'info': f'Unhandled transform type: {transform_type}'}

        transformed_data = registration_processed_df.rdd.map(lambda row: (
            (row.timepoint_id, row.setup_id),
            process_transform(row.transform_type, row.transform_name, row.affine_data)
        )).groupByKey().mapValues(list).collectAsMap()

        return transformed_data
    
    def run(self):
        return self.load_view_registrations()

class ViewInterestPoints():
    def __init__(self, dataframe, base_path_URI):
        self.df = dataframe
        self.base_path_URI = base_path_URI
    
    def load_view_interest_points(self):
        interestPointCollectionLookup = {}
        if self.df.rdd.isEmpty():
            return interestPointCollectionLookup

        interest_points_df = (
            self.df
            .select(
                explode("ViewInterestPoints.ViewInterestPointsFile").alias("interest_points"),
            )
            .select(
                col("interest_points._timepoint").alias("timepoint_id"),
                col("interest_points._setup").alias("setup_id"),
                col("interest_points._label").alias("label"),
                col("interest_points._params").alias("parameters"),
                col("interest_points._VALUE").alias("interest_point_file_name")
            )
        )

        interest_points = interest_points_df.collect()

        for row in interest_points:
            timepoint_id, setup_id, label, parameters, interest_point_file_name = row
            view_id = (timepoint_id, setup_id)

            if view_id not in interestPointCollectionLookup:
                interestPointCollectionLookup[view_id] = {
                    'timepoint_id': timepoint_id,
                    'setup_id': setup_id,
                    'interest_point_lists': {}
                }

            collection = interestPointCollectionLookup[view_id]

            interest_point_list = {
                'base_dir': self.base_path_URI,  
                'modified_interest_points': False,
                'modified_corresponding_interest_points': False,
            }

            if interest_point_file_name.startswith("interestpoints/"):
                interest_point_list.update({
                    'file': interest_point_file_name,
                    'interest_points': None,
                    'corresponding_interest_points': None,
                    'parameters': '',
                    'object_parameters': parameters
                })

            elif interest_point_file_name.startswith("tpId_"):
                interest_point_list.update({
                    'n5path': interest_point_file_name,
                    'object_parameters': parameters
                })

            else:
                raise Exception("Unknown interest point file format.")
            
            collection['interest_point_lists'][label] = interest_point_list

        return interestPointCollectionLookup
    
    def run(self):
        return self.load_view_interest_points()

class BoundingBoxes():
    def __init__(self, dataframe):
        self.df = dataframe
    
    def load_bounding_boxes(self):
        if self.df.rdd.isEmpty() or "ViewRegistrations.ViewRegistration.bounding_boxes" not in self.df.columns:
            return []

        bounding_boxes_df = self.df.select(
            explode("ViewRegistrations.ViewRegistration.bounding_boxes").alias("bounding_box")
        ).select(
            col("bounding_box.title").alias("title"),
            col("bounding_box.min").alias("min"),
            col("bounding_box.max").alias("max")
        )

        bounding_boxes = []
        for row in bounding_boxes_df.collect():
            title = row['title']
            try:
                min_coords = [int(x) for x in row['min'].split()]
                max_coords = [int(x) for x in row['max'].split()]
            except ValueError as e:
                raise Exception(f"Error parsing bounding box coordinates: {str(e)}")

            bounding_boxes.append({
                'title': title,
                'min': min_coords,
                'max': max_coords
            })

        return bounding_boxes
    
    def run(self):
        return self.load_bounding_boxes()

class PointSpreadFunctions():
    def __init__(self, dataframe):
        self.df = dataframe
    
    def load_point_spread_functions(self):
        if "PointSpreadFunctions.PointSpreadFunction" not in self.df.columns: 
            return {}
        
        try:
            psfs_df = self.df.select(
                explode("PointSpreadFunctions.PointSpreadFunction").alias("psf")
            )

            psfs = {}
            for row in psfs_df.collect():
                tp_id = int(row.psf._timepoint) if row.psf._timepoint else None
                vs_id = int(row.psf._setup) if row.psf._setup else None
                file = row.psf._file if row.psf._file else None

                if tp_id is not None and vs_id is not None:
                    view_id = (tp_id, vs_id)

                    psf_entry = {
                        'base_path_URI': self.base_path_URI,
                        'file': file
                    }

                    if view_id not in psfs:
                        psfs[view_id] = []
                    psfs[view_id].append(psf_entry)

            return psfs
        except AnalysisException as e:
            return {}
        
    def run(self):
        return self.load_point_spread_functions()

class StitchingResults():
    def __init__(self, dataframe):
        self.df = dataframe
    
    def load_stitching_results(self):
        if "StitchingResults" not in self.df.columns: return {}
        stitching_df = self.df.select(explode("StitchingResults").alias("pairwiseResult"))

        stitching_results = {}
        for row in stitching_df.collect():
            vsA = list(map(int, row.pairwiseResult.vsA.split(',')))
            vsB = list(map(int, row.pairwiseResult.vsB.split(',')))
            tpA = list(map(int, row.pairwiseResult.tpA.split(',')))
            tpB = list(map(int, row.pairwiseResult.tpB.split(',')))

            shift = np.array(row.pairwiseResult.shift.split()).astype(float)
            corr = float(row.pairwiseResult.corr)
            hash_value = float(row.pairwiseResult.hash) if row.pairwiseResult.hash else 0.0

            transform = np.identity(4)  
            if len(shift) == 3: 
                transform[:3, 3] = shift
            elif len(shift) == 12: 
                transform = np.array(shift).reshape((3, 4))

            vidsA = [(tpA[i], vsA[i]) for i in range(len(vsA))]
            vidsB = [(tpB[i], vsB[i]) for i in range(len(vsB))]

            bbox = None
            if 'bbox' in row.pairwiseResult and row.pairwiseResult.bbox:
                minmax = np.array(row.pairwiseResult.bbox.split()).astype(float)
                bbox = {'min': minmax[:len(minmax)//2], 'max': minmax[len(minmax)//2:]}

            pair_key = (tuple(vidsA), tuple(vidsB))
            stitching_results[pair_key] = {
                'transform': transform,
                'correlation': corr,
                'hash': hash_value,
                'bounding_box': bbox
            }

        return stitching_results
    
    def run(self):
        return self.load_stitching_results

class IntensityAdjustments():
    def __init__(self, dataframe):
        self.df = dataframe
    
    def load_intensity_adjustments(self):
        if "IntensityAdjustments" not in self.df.columns: return {}

        try:
            adjustments_df = self.df.select(explode("IntensityAdjustments").alias("adjustment"))

            intensity_adjustments = {}
            for row in adjustments_df.collect():
                timepoint_id = int(row.adjustment.timepointId)
                setup_id = int(row.adjustment.setupId)
                model_params = np.array(row.adjustment.model.split()).astype(float)

                if len(model_params) != 2:
                    continue  

                affine_model = {'a': model_params[0], 'b': model_params[1]}
                view_id = (timepoint_id, setup_id)

                intensity_adjustments[view_id] = affine_model

            return intensity_adjustments
        except AnalysisException as e:
            return {}
    
    def run(self):
        return self.load_intensity_adjustments()

if __name__ == "__main__":
    args = getResolvedOptions(sys.argv, ['JOB_NAME', 'BUCKET_NAME', 'XML_FILENAME'])

    input_prep = Initialize(args)
    df = input_prep.load_data()

    sequence_description = SequenceDescription(df, input_prep.bucket_name)
    view_registrations = ViewRegistrations(df)
    view_interest_points = ViewInterestPoints(df, input_prep.base_path_uri)
    bounding_boxes = BoundingBoxes(df)
    point_spread_functions = PointSpreadFunctions(df)
    stitching_results = StitchingResults(df)
    intensity_adjustments = IntensityAdjustments(df)
    
    from_sequence_description = sequence_description.run()
    from_view_registrations = view_registrations.run()
    from_view_interest_points = view_interest_points.run()
    from_bounding_boxes = bounding_boxes.run()
    from_point_spread_functions = point_spread_functions.run()
    from_stitching_results = stitching_results.run()
    from_intensity_adjustments = intensity_adjustments.run()

    spim_data = {
        'base_path_URI' : input_prep.base_path_uri,
        'sequence_description': from_sequence_description,
        'view_registrations' : from_view_registrations,
        'view_interest_points' : from_view_interest_points,
        'bounding_boxes' : from_bounding_boxes,                     
        'point_spread_functions' : from_point_spread_functions,      
        'stitching_results' : from_stitching_results,               
        'intensity_adjustments' : from_intensity_adjustments    
    }

    if input_prep.job:
        input_prep.job.commit()
