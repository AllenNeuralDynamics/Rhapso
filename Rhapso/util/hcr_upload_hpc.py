# """This example demonstrates how to submit an HCR job."""

# import json
# import os
# from datetime import datetime
# from glob import glob

# import requests
# from aind_data_schema_models.modalities import Modality
# from aind_data_schema_models.platforms import Platform
# import re
# from aind_codeocean_pipeline_monitor.models import (
#     PipelineMonitorSettings,
#     CaptureSettings,
#     Permissions)
# from codeocean.computation import (
#     DataAssetsRunParam, RunParams)
# from codeocean.data_asset import DataAssetParams

# from aind_data_transfer_service.models.core import (
#     SubmitJobRequestV2,
#     Task,
#     UploadJobConfigsV2,
# )
# from aind_data_schema_models.data_name_patterns import DataLevel
# from aind_data_transfer_models.core import (
#     CodeOceanPipelineMonitorConfigs,
# )
# import argparse


# PROJECT_EMAIL_MAP = {
#     "PLACE": "tim.wang@alleninstitute.org",
#     "MSMA Platform": "kevin.cao@alleninstitute.org",
#     "Learning mFISH-V1omFISH": "marinag@alleninstitute.org",
#     None: "carson.berry@alleninstitute.org",
# }


# def main():
#     """Main function to submit an HCR job."""
#     parser = argparse.ArgumentParser(description="Submit an HCR job to the AIND Data Transfer Service.")
#     parser.add_argument('--project_name', type=str, default="Learning mFISH-V1omFISH", help="Name of the project.")
#     parser.add_argument('--dataset_path', type=str, help="Path to the dataset folder containing HCR data.")
#     parser.add_argument('--subject_id', type=str, default=None, help= "Subject ID for the HCR job. if not provided it will be generated based on the dataset path")
#     parser.add_argument('--user_email', type=str, default=None, help="Email address for job notifications. If not provided, derived from project_name.")


#     args = parser.parse_args()

#     project_name = args.project_name
#     dataset_path = args.dataset_path
#     subject_id = args.subject_id
#     user_email = args.user_email or PROJECT_EMAIL_MAP.get(project_name, PROJECT_EMAIL_MAP[None])
#     if subject_id is None:
#         basename = os.path.basename(dataset_path)
#         if '_' in basename:
#             subject_id = basename.split('_')[1] if 'HCR' in basename else basename.split('_')[0]
#         else:
#             subject_id = basename
#     if not dataset_path:
#         raise ValueError("Please provide the path to the dataset folder containing HCR data using --dataset_path argument.")

#     upload_hcr_job(subject_id, dataset_path, project_name, user_email)

# def get_first_file_size_in_mb(hcr_source: str) -> float:
#     """
#     Get the size of the first file in the SPIM folder under the given hcr_source path.

#     Parameters
#     ----------
#     hcr_source : str
#         The path to the dataset folder containing HCR data.

#     Returns
#     -------
#     float
#         The size of the first file in the SPIM folder in megabytes (MB).

#     Raises
#     ------
#     FileNotFoundError
#         If the SPIM folder does not exist or contains no files.
#     """
#     spim_folder = os.path.join(hcr_source, 'SPIM')

#     if not os.path.exists(spim_folder):
#         raise FileNotFoundError(f"The SPIM folder does not exist at path: {spim_folder}")

#     files = [f for f in os.listdir(spim_folder) if os.path.isfile(os.path.join(spim_folder, f))]

#     if not files:
#         raise FileNotFoundError(f"No files found in the SPIM folder at path: {spim_folder}")

#     first_file_path = os.path.join(spim_folder, files[0])
#     file_size_bytes = os.path.getsize(first_file_path)
#     file_size_mb = file_size_bytes / (1024 * 1024)  # Convert bytes to megabytes

#     return file_size_mb

# def upload_hcr_job(subject_id: str, hcr_source: str, project_name: str, user_email: str = "None") -> None:
#     """This function uploads an HCR job to the AIND Data Transfer Service.

#     Parameters
#     ----------
#     subject_id : str
#         The subject ID for the HCR job.
#     hcr_source : str
#         The path to the dataset folder containing HCR data.
#     project_name : str
#         The name of the project for which the HCR job is being submitted.
#     user_email : str, optional
#         Email address for job notifications, by default "None".

#     Raises
#     ------
#     ValueError
#         If the datetime pattern is not found in the source folder.

#     Notes
#     -----
#     This function retrieves metadata from the AIND Metadata Service and creates empty JSON files if the metadata retrieval fails.
#     It also sets up tasks for processing SPIM data and triggers a CodeOcean pipeline for further processing.
#     """


#     job_type = "HCR"
#     s3_bucket = 'open' # for aind-open-data-dev-u5u0i5

#     #RELEASED PIPELINE
#     CODEOCEAN_PIPELINE_ID = "0eb26d14-b31a-49f7-a347-d028242bd79a" # main pipeline "1042a5cf-b8ae-4dab-b60c-c562d242ddc4" #HCR Version #"7359ac33-3ae2-4bf2-a9c4-d92a315bd62a" # Proteomics pipeline v2
#     CODEOCEAN_PIPELINE_VERSION = 2


#     match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", hcr_source)
#     if match:
#         date_part = match.group(1)
#         time_part = match.group(2).replace("-", ":")
#         acq_datetime = datetime.fromisoformat(f"{date_part}T{time_part}")
#     else:
#         raise ValueError("Datetime pattern not found in source_folder.")

#     labtracks_id = subject_id.split("-")[0]

#     # Download subject and procedures from metadata service
#     metadata_service_url = "http://aind-metadata-service"  # for prod
#     # metadata_service_url = "http://aind-metadata-service-dev"  # for testing
#     metadata_files = [os.path.basename(x) for x in glob(f"{hcr_source}/*.json")]

#     # Create empty subject.json if metadata service fails or for testing
#     if "subject.json" not in metadata_files:
#         try:
#             subject_response = requests.get(
#                 f"{metadata_service_url}/subject/{labtracks_id}"
#             )
#             if subject_response.status_code in [200, 406]:
#                 json_data = subject_response.json()["data"]
#                 with open(f"{hcr_source}/subject.json", "w") as f:
#                     json.dump(json_data, f, indent=3)
#             else:
#                 subject_response.raise_for_status()
#         except Exception as e:
#             print(f"Failed to get subject from metadata service: {e}")
#             print("Creating empty subject.json file...")

#             empty_subject = {
#                 "describedBy": "https://raw.githubusercontent.com/AllenNeuralDynamics/aind-data-schema/main/src/aind_data_schema/core/subject.py",
#                 "schema_version": "1.0.0",
#                 "subject_id": subject_id,
#                 "sex": "Unknown",
#                 "date_of_birth": "2025-01-01",
#                 "genotype": "Unknown",
#                 "species": {
#                     "name": None,
#                     "abbreviation": None,
#                     "registry": {
#                         "name": "National Center for Biotechnology Information",
#                         "abbreviation": "NCBI"
#                     },
#                     "registry_identifier": "10090"
#                 },
#                 "alleles": [],
#                 "background_strain": None,
#                 "breeding_info": [],
#                 "housing": None,
#                 "notes": None,
#                 "wellness_reports": [],
#                 "restrictions": None
#             }

#             with open(f"{hcr_source}/subject.json", "w") as f:
#                 json.dump(empty_subject, f, indent=3)

#     # Create empty procedures.json if metadata service fails or for testing
#     if "procedures.json" not in metadata_files:
#         try:
#             procedures_response = requests.get(
#                 f"{metadata_service_url}/procedures/{labtracks_id}"
#             )
#             if procedures_response.status_code in [200, 406]:
#                 json_data = procedures_response.json()["data"]
#                 with open(f"{hcr_source}/procedures.json", "w") as f:
#                     json.dump(json_data, f, indent=3)
#             else:
#                 procedures_response.raise_for_status()
#         except Exception as e:
#             print(f"Failed to get procedures from metadata service: {e}")
#             print("Creating empty procedures.json file...")

#             empty_procedures = {
#                 "describedBy": "https://raw.githubusercontent.com/AllenNeuralDynamics/aind-data-schema/main/src/aind_data_schema/core/procedures.py",
#                 "schema_version": "1.0.0",
#                 "subject_id": subject_id,
#                 "subject_procedures": [],
#                 "notes": None
#             }

#             with open(f"{hcr_source}/procedures.json", "w") as f:
#                 json.dump(empty_procedures, f, indent=3)

#     number_of_tiles = len(glob(f"{hcr_source}/SPIM/*.czi"))
#     MIN_NUMBER_OF_PARTITIONS = 32
#     num_of_partitions: int = min(number_of_tiles, MIN_NUMBER_OF_PARTITIONS)

#     number_tiles_per_partition = number_of_tiles // num_of_partitions
#     memory_overhead_from_scheduling_tiles = number_tiles_per_partition*1300 # in MB, 1300MB is the overhead for scheduling a tile (recorded so far)
#     memory_overhead_from_processing_shard = 4400 # in MB, 4400MB is the overhead for processing a shard (recorded so far)
#     buffer_memory = 1000 # in MB, buffer memory to avoid OOM errors


#     PEAK_MEMORY_SCALE_FACTOR = 2 # Scale factor for memory allocation
#     CPUS_PER_NODE = 1  # Number of CPUs per node

#     estimated_peak_memory_per_cpu = int((
#         memory_overhead_from_scheduling_tiles +
#         memory_overhead_from_processing_shard +
#         buffer_memory
#         )/CPUS_PER_NODE)  # in MB

#     #get the size of the first file in the SPIM folder
#     tile_size_mb = get_first_file_size_in_mb(hcr_source)
#     # MIN_RAM_PER_NODE = 30000  # Minimum RAM per node in MB
#     # MAX_RAM_PER_NODE = 48000  # Maximum RAM per node in MB
#     MIN_RAM_PER_NODE = 64000  # Minimum RAM per node in MB
#     MAX_RAM_PER_NODE = 64000  # Maximum RAM per node in MB
#     tile_size_estimated_memory = int(tile_size_mb * PEAK_MEMORY_SCALE_FACTOR/ CPUS_PER_NODE)
#     PEAK_MEMORY_PER_CPU = min(max(estimated_peak_memory_per_cpu, int(MIN_RAM_PER_NODE/CPUS_PER_NODE)), int(MAX_RAM_PER_NODE/CPUS_PER_NODE))  # in MB, min 6GB per CPU, max 10GB per CPU


#     #upload runtime timeout to be the estimated time to process the data + 1 hour buffer
#     estimated_time_to_process_data_in_minutes= int((number_of_tiles * tile_size_mb / (1024))/4200 + 60)*8 # 4200 MB/hour is the processing speed

#     print(f'Tile size in MB: {tile_size_mb}')
#     print(f'Peak memory in MB: {tile_size_mb * PEAK_MEMORY_SCALE_FACTOR}')
#     print(f'Resources Requested: {num_of_partitions} partitions, \
#           {CPUS_PER_NODE} CPUs per node, \
#           {PEAK_MEMORY_PER_CPU} MB per CPU')
#     print(f'Timeout Limit Requested: {estimated_time_to_process_data_in_minutes}')


#     spim_task = Task(
#         image_resources=(
#             {
#                 # "array": f"0-{num_of_partitions - 1}",
#                 "array": "10,21",
#                 "time_limit": {"set": True, "number": estimated_time_to_process_data_in_minutes},  # 3 hours
#                 "memory_per_cpu": {"set": True, "number": PEAK_MEMORY_PER_CPU},
#                 "minimum_cpus_per_node": CPUS_PER_NODE,
#                 # "comment": "retry 2", # to retry 2 times on failure
#                 "comment": "retry p010,p021 cpu1 mem64g",
#             }
#         ),
#         image="ghcr.io/allenneuraldynamics/aind-hcr-data-transformation",
#         image_version= "dev-cf71aa2", #"dev-68cbeba",#"0.0.7", #"dev-bd73522", #TODO point to the latest dev version - first try always fails becaues they have to download it for the first time...
#         job_settings={
#             "input_source": hcr_source,
#             "num_of_partitions": num_of_partitions,
#         },
#     )

#     modality_settings = {
#         Modality.SPIM.abbreviation: spim_task,
#     }

#     gather_preliminary_metadata = Task(job_settings={"metadata_dir": hcr_source})

#     codeocean_pipeline_settings_spim = PipelineMonitorSettings(
#         run_params=RunParams(
#             pipeline_id=CODEOCEAN_PIPELINE_ID,
#             data_assets=[
#                 DataAssetsRunParam(id="", mount="proteomics_dataset")
#                 ],
#             version = CODEOCEAN_PIPELINE_VERSION,
#             ),
#         capture_settings = CaptureSettings(
#             # permissions= Permissions(), #default is everyone can read, data admin owns
#             tags = ["HCR", "RAW"],
#         ),
#         computation_timeout=100.0 * 60 * 60,  # 100 hour timeout
#     )


#     trigger_co_pipeline_settings = {
#         Modality.SPIM.abbreviation: {
#             "skip_task": False,
#             "job_settings": {
#                 "pipeline_monitor_settings": (
#                     codeocean_pipeline_settings_spim.model_dump(
#                         mode="json", exclude_none=True
#                     )
#                 )
#             },
#         }
#     }

#     # The job_type loads defaults settings from AWS Parameter Store
#     upload_job_configs_v2 = UploadJobConfigsV2(
#         job_type=job_type,
#         user_email=user_email,
#         email_notification_types = ["all"],
#         s3_bucket=s3_bucket,
#         project_name=project_name,
#         platform=Platform.HCR,
#         modalities=[Modality.SPIM],
#         subject_id=subject_id,
#         acq_datetime=acq_datetime,
#         tasks={
#             "modality_transformation_settings": modality_settings,
#             "check_s3_folder_exists": {"skip_task": False},
#             "final_check_s3_folder_exist": {"skip_task": True},
#             "check_metadata_files": {"skip_task": False},
#             "gather_preliminary_metadata": gather_preliminary_metadata,
#             "codeocean_pipeline_settings": trigger_co_pipeline_settings,
#         },
#     )

#     upload_jobs = [upload_job_configs_v2]

#     submit_request = SubmitJobRequestV2(
#         upload_jobs=upload_jobs,
#     )

#     post_request_content = submit_request.model_dump(
#         mode="json", exclude_none=True
#     )

#     # Please use the production endpoint for submitting jobs and the dev endpoint
#     # for running tests.
#     endpoint = "http://aind-data-transfer-service"  # For production
#     # endpoint = "http://aind-data-transfer-service-dev"  # For testing

#     submit_job_response = requests.post(
#         url=f"{endpoint}/api/v2/submit_jobs",
#         json=post_request_content,
#     )
#     print(submit_job_response.status_code)
#     print(submit_job_response.json())
#     print(f'Review your job at {endpoint}/jobs/')



# if __name__ == "__main__":
#     main()
# (base) [svc_aind_imaging@hpc hcr-czitile-utils-editable]$ 