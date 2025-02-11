import boto3
import json
import os
import zipfile
from io import BytesIO
import subprocess
import pandas as pd
from io import StringIO
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

class SparkPipeline():
    def __init__(self, dataframes, overlapping_area, dsxy, dsz, prefix, **kwargs):
        self.dataframes = dataframes
        self.overlapping_area = overlapping_area
        self.dsxy = dsxy
        self.dsz = dsz
        self.prefix = prefix
        self.bucket_name = kwargs.get('bucket_name')
        self.region = kwargs.get('region')
        self.job_name = kwargs.get('job_name')
        self.role = kwargs.get('role')
        self.worker_type = kwargs.get('worker_type')
        self.number_of_workers = kwargs.get('number_of_workers')
        self.glue_version = kwargs.get('glue_version')
        self.job_timeout = kwargs.get('job_timeout')
        self.retries = kwargs.get('retries')
        self.job_bookmark_option = kwargs.get('job_bookmark_option')
        self.flex_execution = kwargs.get('flex_execution')
        self.rhapso_prefix = kwargs.get('rhapso_prefix') + '/interest_point_detection/spark_etl/interest_point_detection.py'
        
        self.s3 = boto3.client('s3')
        self.cloud_script_location = f's3://{self.bucket_name}/interest_point_detection.py'

    def create_s3_bucket(self):
        try:
            self.s3.create_bucket(Bucket=self.bucket_name, CreateBucketConfiguration={
                'LocationConstraint': self.region
            })
        except (NoCredentialsError, PartialCredentialsError) as e:
            print("Credentials problem: Check your AWS setup.", e)
        except self.s3.exceptions.BucketAlreadyOwnedByYou:
            print(f"Bucket '{self.bucket_name}' already exists and is owned by you.")
        except Exception as e:
            print("Failed to create bucket:", e)
    
    # this is only for dev work, rhapso package will replace this func
    def serialize_import_packages_to_s3(self):
        subprocess.run(['python', '-m', 'venv', 'rhapso-viv'])
        subprocess.run([f"./rhapso-viv/bin/pip", 'install', 'numpy', 'dask[array]', 'matplotlib', 'bioio', 'bioio-tifffile'], check=True)
        
        site_packages_path = f"./rhapso-viv/lib/python3.11/site-packages" 

        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(site_packages_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=os.path.abspath(site_packages_path))
                    zipf.write(file_path, arcname)  
        
        zip_buffer.seek(0)
        object_key = 'import_packages/zip_buffer'
        self.s3.upload_fileobj(zip_buffer, self.bucket_name, object_key)
    
    def serialize_bounds_to_s3(self):
        try:
            rows = []
            for key, bounds in self.overlapping_area.items():
                timepoint, setup = key.split(', ')
                for bound in bounds:
                    rows.append({
                        'timepoint': timepoint.split(': ')[1],
                        'setup': setup.split(': ')[1],
                        'lower_bound_x': bound['lower_bound'][0],
                        'lower_bound_y': bound['lower_bound'][1],
                        'lower_bound_z': bound['lower_bound'][2],
                        'upper_bound_x': bound['upper_bound'][0],
                        'upper_bound_y': bound['upper_bound'][1],
                        'upper_bound_z': bound['upper_bound'][2],
                        'span_x': bound['span'][0],
                        'span_y': bound['span'][1],
                        'span_z': bound['span'][2]
                    })
            df = pd.DataFrame(rows)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            object_key = 'dataframes/bounds.csv'
            self.s3.put_object(Bucket=self.bucket_name, Key=object_key, Body=csv_buffer.getvalue())
        except Exception as e:
            print(f'Failed to upload DataFrame bounds to S3: {str(e)}')
    
    def serialize_dataframes_to_s3(self):
        try:
            for name, df in self.dataframes.items():
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                object_key = f"dataframes/{name}.csv"
                self.s3.put_object(Bucket=self.bucket_name, Key=object_key, Body=csv_buffer.getvalue())
        except Exception as e:
            print(f"Failed to upload DataFrame {name} to S3: {str(e)}")

    def serialize_params_to_s3(self): 
        try:     
            params_data = {
                'dsxy': self.dsxy,
                'dsz': self.dsz,
                'prefix': self.prefix,
            }
            object_key = "params"
            params_json = json.dumps(params_data)
            self.s3.put_object(Bucket=self.bucket_name, Key=object_key, Body=params_json)
        except Exception as e:
            print(f"Failed to serialize or upload parameters to S3: {str(e)}")

    def upload_python_script_to_s3(self, path):
        object_name = path.split('/')[-1]
        try:
            self.s3.upload_file(path, self.bucket_name, object_name)
        except FileNotFoundError:
            print(f"The file '{path}' was not found.")
        except NoCredentialsError:
            print("No credentials available to access AWS services.")

    def create_and_run_glue_job(self):
        glue = boto3.client('glue', region_name=self.region)
        try:
            job = glue.create_job(
                Name=self.job_name,
                Role=self.role,
                Command={
                    'Name': 'glueetl',
                    'ScriptLocation': self.cloud_script_location,
                    'PythonVersion': '3'
                },
                GlueVersion = self.glue_version,
                WorkerType=self.worker_type,
                NumberOfWorkers=self.number_of_workers,
                ExecutionProperty={
                    'MaxConcurrentRuns': 1
                },
                DefaultArguments={
                    '--job-bookmark-option': self.job_bookmark_option,
                    '--enable-s3-parquet-optimized-committer': 'true',
                    '--job-timeout': str(self.job_timeout * 60),  
                    '--retry': str(self.retries),
                    '--flex_execution': str(self.flex_execution).lower(),
                    '--bucket_name': self.bucket_name
                }
            )
            print(f"Glue job '{self.job_name}' created.")
            start_job = glue.start_job_run(JobName=self.job_name)
            print(f"Glue job '{self.job_name}' started with run ID: {start_job['JobRunId']}")
        except Exception as e:
            print("Failed to create or run Glue job:", e)
    
    # TODO -implement
    def get_output_from_s3():
        print("hi")

    def run(self):
        self.create_s3_bucket() 
        self.serialize_import_packages_to_s3()
        self.serialize_bounds_to_s3()
        self.serialize_dataframes_to_s3()
        self.serialize_params_to_s3()
        self.upload_python_script_to_s3(self.rhapso_prefix)
        self.create_and_run_glue_job()
        # self.get_output_from_s3()