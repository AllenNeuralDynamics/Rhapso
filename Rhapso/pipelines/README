# Rhapso Pipeline Scripts README

## Introduction

This README outlines template pipeline scripts that run Rhapso in different environments, stages, optimizations, and data locations. Input parameters are xml data and image data, locally or from S3. Output is an accurate 3D image of the dataset.

## Rhapso Components

**Interest Point Detection**: Implements the Difference of Gaussian (DoG) algorithm on prepared image data to identify points of interest. It accepts zarr or tiff image data along with one XML file as input and produces one XML file as output.

**Interest Point Matching**: 

**Solve**: 

**Fusion**: 

## End-to-End Pipelines

The end-to-end pipelines are designed to run the Rhapso process from start to finish. Please note that these pipelines are shared resources. It's best practice to develop using the development pipelines and discuss before integrating with end-to-end.

### Python Pipeline

**Location**: `/pipelines/python_pipeline.py`

- **Description**: This pipeline uses the core end-to-end Rhapso Python components. 

**Execution**: To run this pipeline, set your launch.json module attribute to `Rhapso.pipelines.python_pipeline` and execute the script with Python. Ensure all dependencies are installed and AWS credentials are configured if using S3 for input/output locations.

**Optimization**: This pipeline includes a Dask optimization option in the IPD Difference of Gaussian (DoG) class, suitable for single-thread interpreters only. Uncomment the desired version to activate it.

### Spark ETL Pipeline

- **Location**: `/pipelines/spark_etl_pipeline.py`

- **Description**: This pipeline uses the core end-to-end Rhapso Python components within AWS Glue (Spark-ETL)

**Execution**: To operate this pipeline, copy it into an AWS Glue job (Spark) and set the version to 5.0. Under advanced settings, include the parameter --additional-python-modules and specify the S3 location of the Rhapso whl file (for development use only). Ensure you assign unique names to the Glue Crawler and Glue Database to avoid overwriting others' work. This pipeline exclusively handles tiff image data, with both input and output based solely on S3.

temp whl file location: `s3://rhapso-whl/Rhapso-0.1.0-py3-none-any.whl`

**Optimization**: This pipeline uses the most optimized version of Rhapso for handling very large datasets. To reduce costs, a hybrid Spark-ETL pipeline is available. Further optimization is potentially achievable by developing custom data handlers that flatten and serialize the image data. Currently, image data is staged in Parquet format, with each file sized between 512 MB and 1 GB. Each image chunk is approximately 20 MB, with six chunks per partition (120 MB), and eight partitions per Parquet file (1 GB).

## Development-Specific Directories

These directories are for individual development and testing.

### IPD Development

- **Location**: `/pipelines/ipd_dev/`

- **Content**: This setup includes both Python and Spark ETL pipelines designed specifically for interest point detection (IPD). These scripts are intended for development purposes and are not end-to-end. It also features a hybrid_prep pipeline to flatten and serialize image data within the Python interpreter.

- **Execution**: To run this pipeline, set your launch.json module attribute to `Rhapso.pipelines.ipd_dev.python_pipeline` or `Rhapso.pipelines.ipd_dev.spark-etl_pipeline` or `Rhapso.pipelines.ipd_dev.hybrid_pipeline_prep`

### IPM Development

### SOLVE Development

### FUSION Development

