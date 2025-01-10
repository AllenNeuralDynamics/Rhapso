import sys
import os
import boto3
import json
import numpy as np
import pandas as pd
from pyspark.sql import Row, SparkSession, DataFrame
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.sql.utils import AnalysisException
from pyspark.sql.functions import explode, col
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType, FloatType, ArrayType, MapType
from pyspark.sql.functions import col, explode, udf, collect_list, udf

# This version is designed to run locally

# Pull and Run Docker Image: 
# docker pull amazon/aws-glue-libs:glue_libs_4.0.0_image_01
# docker run -it --rm amazon/aws-glue-libs:glue_libs_4.0.0_image_01

# AWS auth inside docker image:
# aws configure

# To Run Script
# python3 /jupyter_workspace/glue/input_preparation_local.py --JOB_NAME input_preparation_local --BUCKET_NAME rhapso-example-data-zarr --XML_FILENAME dataset.xml

class Initialize:
    def __init__(self, args):
        self.job_name = args['JOB_NAME']
        self.bucket_name = args['BUCKET_NAME']
        self.xml_filename = args['XML_FILENAME']
        self.s3_input_path = f"s3://{self.bucket_name}/{self.xml_filename}"
        self.output_path = f"s3://{self.bucket_name}/spim-data/"
        self.setup_contexts()
        self.base_path_uri = self.base_path_URI(self.s3_input_path)
        self.df = None
    
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
    
    def base_path_URI(self, file_key: str):
        directory_path = os.path.dirname(file_key)
        base_path_uri = f"s3://{self.bucket_name}/{directory_path}" if directory_path else f"s3://{self.bucket_name}/"
        
        schema = StructType([StructField("base_path_uri", StringType(), True)])
        uri_df = self.spark.createDataFrame([(base_path_uri,)], schema)
        
        return uri_df

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
    def __init__(self, dataframe, bucket_name, spark):
        self.df = dataframe
        self.bucket_name = bucket_name
        self.spark = spark
     
    def load_view_setups(self):
        try:
            exploded_df = self.df.select(explode("SequenceDescription.ViewSetups.ViewSetup").alias("view_setup"))   
            if exploded_df.rdd.isEmpty(): raise ValueError("No view setups found in DataFrame.")
            
            view_setups_list = [
                {**row.view_setup.asDict(), 'id': int(row.view_setup.id)}
                for row in exploded_df.collect()
                if row.view_setup and row.view_setup.id is not None
            ]

            if not view_setups_list:
                raise ValueError("View setups list is empty after processing.")
            
            view_setups_df = self.spark.createDataFrame(view_setups_list)

            return view_setups_df
        
        except AnalysisException as e:
            raise ValueError(f"Error processing DataFrame: {e}")
    
    def load_time_points(self):
        try:
            timepoints_df = self.df.select("SequenceDescription.Timepoints.*")
            if timepoints_df.rdd.isEmpty(): raise ValueError("Timepoints data is missing in the DataFrame.")
            
            rows = []
            timepoints_type = timepoints_df.first()._type

            if timepoints_type == "pattern":
                pattern = timepoints_df.first().integerpattern
                rows.append(Row(type='pattern', data=pattern))

            elif timepoints_type == "range":
                first = timepoints_df.first().first
                last = timepoints_df.first().last
                rows.extend(Row(type='range', timepoint=t) for t in range(first, last + 1))

            elif timepoints_type == "list":
                list_ids = timepoints_df.select(explode("id").alias("id"))
                rows.extend(Row(type='list', timepoint=row.id) for row in list_ids.collect())

            else:
                raise ValueError(f"Unknown timepoints type: {timepoints_type}")
            
            result_df = self.spark.createDataFrame(rows)

            return result_df

        except AnalysisException as e:
            raise ValueError(f"Error accessing timepoints data: {e}")

    def load_missing_views(self):
        if 'MissingViews' in self.df.columns:
            try:
                missing_views_df = self.df.select(explode("SequenceDescription.MissingViews.view").alias("missing_view"))
                if missing_views_df.rdd.isEmpty():
                    return self.spark.createDataFrame([], schema="timepoint INT, setup INT")

                rows = [
                    Row(timepoint=row.missing_view.timepoint, setup=row.missing_view.setup)
                    for row in missing_views_df.collect()
                    if row.missing_view is not None
                ]

                result_df = self.spark.createDataFrame(rows)

                return result_df
            
            except AnalysisException as e:
                print(f"Error processing DataFrame for missing views: {e}")
                return self.spark.createDataFrame([], schema="timepoint INT, setup INT")
                
        else:
            return self.spark.createDataFrame([], schema="timepoint INT, setup INT")
    
    def zarr_image_loader(self):
        self.spark = SparkSession.builder.appName("Zarr Image Loader").getOrCreate()
        schema = StructType([
            StructField("timepoint", IntegerType(), True),
            StructField("setup", IntegerType(), True),
            StructField("path", StringType(), True),
            StructField("uri", StringType(), True),
            StructField("bucket_name", StringType(), True),
            StructField("s3_mode", BooleanType(), True)
        ])
        
        try:
            s3_bucket_element = self.df.select("SequenceDescription.ImageLoader.s3bucket").first()
            bucket = s3_bucket_element[0] if s3_bucket_element and s3_bucket_element[0] else None
            zarr_folder_element = self.df.select("SequenceDescription.ImageLoader.zarr").first()
            folder = zarr_folder_element[0]._VALUE if zarr_folder_element and hasattr(zarr_folder_element[0], "_VALUE") else str(zarr_folder_element[0]) if zarr_folder_element else ""
            folder = f"/{folder.strip('/')}/"

            uri = f"s3://{bucket}{folder}" if bucket else folder
            zgroups_df = self.df.select(explode("SequenceDescription.ImageLoader.zgroups.zgroup").alias("zgroup"))
            
            rows = [
                Row(timepoint=int(row.zgroup['_timepoint']), setup=int(row.zgroup['_setup']), path=row.zgroup['path'], uri=uri, bucket_name=bucket, s3_mode=bool(bucket))
                for row in zgroups_df.collect()
                if row.zgroup and row.zgroup['path'] is not None
            ]
            
            output = self.spark.createDataFrame(rows, schema)
            output.show(20)
            return output
        
        except Exception as e:
            print(f"❌ Error in Zarr Image Loader: {e}")
            return self.spark.createDataFrame([], schema)
    
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
                return self.zarr_image_loader()
            elif image_loader_format == "spimreconstruction.filemap2":
                return self.tiff_image_loader(sequence_description)
            else:
                raise ValueError(f"Unsupported image format: {image_loader_format}")

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

        transform_udf = udf(process_transform, StructType([
            StructField("type", StringType(), True),
            StructField("name", StringType(), True),
            StructField("affine", ArrayType(ArrayType(FloatType(), False), False), True)
        ]))

        transformed_data = registration_processed_df.withColumn(
            "transformed",
            transform_udf("transform_type", "transform_name", "affine_data")
        )

        grouped_transformed_data = transformed_data.groupBy("timepoint_id", "setup_id").agg(
            collect_list("transformed").alias("transformations")
        )

        return grouped_transformed_data
    
    def run(self):
        return self.load_view_registrations()

class ViewInterestPoints():
    def __init__(self, dataframe, spark, base_path_URI):
        self.df = dataframe
        self.base_path_URI = base_path_URI
        self.spark = spark
    
    def load_view_interest_points(self):
        if self.df.rdd.isEmpty():
            print("DataFrame is empty.")
            return None

        # Explode and select necessary fields
        interest_points_df = (
            self.df
            .select(explode("ViewInterestPoints.ViewInterestPointsFile").alias("interest_points"))
            .select(
                col("interest_points._timepoint").alias("timepoint_id"),
                col("interest_points._setup").alias("setup_id"),
                col("interest_points._label").alias("label"),
                col("interest_points._params").alias("parameters"),
                col("interest_points._VALUE").alias("interest_point_file_name")
            )
        )

        # Dynamically determine schema based on the file name pattern
        def determine_schema(file_name):
            if file_name.startswith("interestpoints/"):
                return ("file", StructType([
                    StructField("timepoint_id", IntegerType(), True),
                    StructField("setup_id", IntegerType(), True),
                    StructField("label", StringType(), True),
                    StructField("parameters", StringType(), True),
                    StructField("file", StringType(), True),
                    StructField("base_dir", StringType(), True),
                    StructField("object_parameters", StringType(), True),
                    StructField("modified_interest_points", BooleanType(), True),
                    StructField("modified_corresponding_interest_points", BooleanType(), True)
                ]))
            elif file_name.startswith("tpId_"):
                return ("n5path", StructType([
                    StructField("timepoint_id", IntegerType(), True),
                    StructField("setup_id", IntegerType(), True),
                    StructField("label", StringType(), True),
                    StructField("parameters", StringType(), True),
                    StructField("n5path", StringType(), True),
                    StructField("base_dir", StringType(), True),
                    StructField("object_parameters", StringType(), True),
                    StructField("modified_interest_points", BooleanType(), True),
                    StructField("modified_corresponding_interest_points", BooleanType(), True)
                ]))
            else:
                return (None, None)

        file_type_udf = udf(determine_schema, StringType())
        final_df = interest_points_df.withColumn("file_type", file_type_udf(col("interest_point_file_name")))

        return final_df
    
    def run(self):
        return self.load_view_interest_points()

class BoundingBoxes():
    def __init__(self, dataframe):
        self.df = dataframe
    
    def load_bounding_boxes(self):
        if self.df.rdd.isEmpty() or "ViewRegistrations.ViewRegistration.bounding_boxes" not in self.df.columns:
            return self.df.sparkSession.createDataFrame([], StructType([]))

        bounding_boxes_df = self.df.select(
            explode("ViewRegistrations.ViewRegistration.bounding_boxes").alias("bounding_box")
        ).select(
            col("bounding_box.title").alias("title"),
            col("bounding_box.min").alias("min"),
            col("bounding_box.max").alias("max")
        )

        results = []
        for row in bounding_boxes_df.collect():
            title = row['title']
            try:
                min_coords = [int(x) for x in row['min'].split()]
                max_coords = [int(x) for x in row['max'].split()]
            except ValueError as e:
                raise Exception(f"Error parsing bounding box coordinates: {str(e)}")

            result = Row(
                title=title,
                min=min_coords,
                max=max_coords
            )
            results.append(result)

        schema = StructType([
            StructField("title", StringType(), True),
            StructField("min", ArrayType(IntegerType()), True),
            StructField("max", ArrayType(IntegerType()), True)
        ])

        return self.df.sparkSession.createDataFrame(results, schema)
    
    def run(self):
        return self.load_bounding_boxes()

class PointSpreadFunctions():
    def __init__(self, dataframe):
        self.df = dataframe
    
    def load_point_spread_functions(self):
        if "PointSpreadFunctions.PointSpreadFunction" not in self.df.columns:
            return self.df.sparkSession.createDataFrame([], StructType([]))
        
        psfs_df = self.df.select(
            explode("PointSpreadFunctions.PointSpreadFunction").alias("psf")
        )

        results = []
        try:
            for row in psfs_df.collect():
                tp_id = int(row.psf._timepoint) if row.psf._timepoint else None
                vs_id = int(row.psf._setup) if row.psf._setup else None
                file = row.psf._file if row.psf._file else None

                if tp_id is not None and vs_id is not None:
                    result = Row(
                        timepoint_id=tp_id,
                        setup_id=vs_id,
                        base_path_URI=self.base_path_URI,
                        file=file
                    )
                    results.append(result)

            schema = StructType([
                StructField("timepoint_id", IntegerType(), True),
                StructField("setup_id", IntegerType(), True),
                StructField("base_path_URI", StringType(), True),
                StructField("file", StringType(), True)
            ])

            return self.df.sparkSession.createDataFrame(results, schema)

        except AnalysisException as e:
            return self.df.sparkSession.createDataFrame([], schema)
        
    def run(self):
        return self.load_point_spread_functions()

class StitchingResults():
    def __init__(self, dataframe):
        self.df = dataframe
    
    def load_stitching_results(self):
        if "StitchingResults" not in self.df.columns:
            return self.df.sparkSession.createDataFrame([], StructType([]))
        
        stitching_df = self.df.select(explode("StitchingResults").alias("pairwiseResult"))
        results = []
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

            result = Row(pairwise_key=(tuple(vidsA), tuple(vidsB)),
                         transform=transform.flatten().tolist(),
                         correlation=corr,
                         hash=hash_value,
                         bounding_box=bbox)

            results.append(result)
        
        schema = StructType([
            StructField("pairwise_key", StringType(), True),
            StructField("transform", ArrayType(FloatType()), True),
            StructField("correlation", FloatType(), True),
            StructField("hash", FloatType(), True),
            StructField("bounding_box", MapType(StringType(), ArrayType(FloatType())), True)
        ])
        
        stitching_df = self.df.sparkSession.createDataFrame(results, schema)
        return stitching_df
    
    def run(self):
        return self.load_stitching_results()

class IntensityAdjustments():
    def __init__(self, dataframe):
        self.df = dataframe
    
    def load_intensity_adjustments(self):
        if "IntensityAdjustments" not in self.df.columns: 
            return self.df.sparkSession.createDataFrame([], StructType([]))

        try:
            adjustments_df = self.df.select(
                col("adjustment.timepointId").cast("int").alias("timepoint_id"),
                col("adjustment.setupId").cast("int").alias("setup_id"),
                col("adjustment.model")
            )

            def parse_model(model_str):
                params = np.array(model_str.split()).astype(float)
                if len(params) == 2:
                    return {'a': params[0], 'b': params[1]}
                else:
                    return None

            parse_model_udf = udf(parse_model, MapType(StringType(), FloatType()))
            result_df = adjustments_df.withColumn("affine_model", parse_model_udf(col("model"))).drop("model")
            
            return result_df

        except Exception as e:
            print(f"Error processing intensity adjustments: {e}")
            return self.df.sparkSession.createDataFrame([], StructType([]))
    
    def run(self):
        return self.load_intensity_adjustments()

if __name__ == "__main__":
    
    def save_to_parquet(df, output_path):
        if df and not df.rdd.isEmpty():
            df.write.mode('overwrite').parquet(output_path)

    def save_nested_dataframes(data, base_path):
        for key, value in data.items():
            current_path = f"{base_path}/{key}"
            if isinstance(value, DataFrame):
                save_to_parquet(value, current_path)
            elif isinstance(value, dict) and key == 'sequence_description':
                for nested_key, nested_value in value.items():
                    nested_path = f"{current_path}/{nested_key}"
                    if isinstance(nested_value, DataFrame):
                        save_to_parquet(nested_value, nested_path)
        
    args = getResolvedOptions(sys.argv, ['JOB_NAME', 'BUCKET_NAME', 'XML_FILENAME'])

    input_prep = Initialize(args)
    df = input_prep.load_data()

    sequence_description = SequenceDescription(df, input_prep.bucket_name, input_prep.spark)
    view_registrations = ViewRegistrations(df)
    view_interest_points = ViewInterestPoints(df, input_prep.spark, input_prep.base_path_uri)
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
        'base_path_uri' : input_prep.base_path_uri,
        'sequence_description': from_sequence_description,
        'view_registrations' : from_view_registrations,
        'view_interest_points' : from_view_interest_points,
        'bounding_boxes' : from_bounding_boxes,                     
        'point_spread_functions' : from_point_spread_functions,      
        'stitching_results' : from_stitching_results,               
        'intensity_adjustments' : from_intensity_adjustments    
    } 

    base_output_path = f"s3://{input_prep.bucket_name}/output-ipd"
    save_nested_dataframes(spim_data, base_output_path)

    if input_prep.job:
        input_prep.job.commit()
