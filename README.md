# Rhapso

**Rhapso** is a modular Python toolkit for aligning and fusing large-scale microscopy datasets. 

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/AllenNeuralDynamics/Rhapso/ci.yml?branch=main)](https://github.com/AllenNeuralDynamics/Rhapso/actions)
[![Documentation](https://img.shields.io/badge/docs-wiki-blue)](https://github.com/AllenNeuralDynamics/Rhapso/wiki)
[![Issues](https://img.shields.io/github/issues/AllenNeuralDynamics/Rhapso)](https://github.com/AllenNeuralDynamics/Rhapso/issues)

<!-- ## Example Usage Media Content Coming Soon....
-- -->

## Connect With Us
<!-- UPDATE THIS WHEN OPEN SOURCED -->
  [Allen Institute Internal Coms](https://teams.microsoft.com/l/channel/19%3AIv_3CdryfboX3E0g1BbzCi3Y8KRgNCdAv3idJ9epV-k1%40thread.tacv2/Project%20Rhapso-Shared?groupId=87b91b48-bb2a-4a00-bc59-5245043a0708&tenantId=32669cd6-737f-4b39-8bdd-d6951120d3fc&ngc=true&allowXTenantAccess=true)

## Table of Contents
- [Summary](#summary)
- [Build Package](#build-package)
  - [Using the Built `.whl` File](#using-the-built-whl-file)
- [Usage](#usage)
- [Componenents Walkthrough](#components-walkthrough)
- [Command Line Usage](#command-line-usage)
- [Setup Instructions](#setup-instructions)
- [Run Tests](#run-tests)
- [Environments](#environments)
- [Use Cases](#use-cases)
- [FAQ](#frequently-asked-questions)

---

## Summary
Built from decoupled components, Rhapso separates data loading and execution logic from its core functionality, making it flexible and easy to adapt across environments and formats.

To run Rhapso, users can either provide a data loader and a pipeline script that orchestrates the processing steps or use one of ours. We include example loaders and pipeline scripts to support both large-scale runs on AWS Glue (Spark ETL) and smaller-scale testing on local or conventional machines. Input formats like OME-TIFF and Zarr are supported out of the box.

Rhapso is developed in collaboration with the Allen Institute for Neural Dynamics (AIND), initially supporting AIND’s ExaSPIM pipeline and eventually broadly benefiting microscopy research.

### Environments
This package is designed to target three main environments:
- Local
- Cloud
- SLURM
> [!TIP]
> Detailed instructions on how to run a sample end-to-end pipeline for each environment can be found on the [Wiki Page](https://github.com/AllenNeuralDynamics/Rhapso/wiki#example-pipelines).
Sample pipeline instructions are provided for pre-made templates, but if you want to create your own template, you can do so based on the sample files available inside the `Rhapso/pipelines` folder.

---

## Getting Started

### Clone Rhapso

 ```sh
  git clone https://github.com/AllenNeuralDynamics/Rhapso.git
  ```
### Clone Rhapso Wiki
 ```sh
  git clone https://github.com/AllenNeuralDynamics/Rhapso.wiki.git
  ```
### Downkoad and Install Dependencies
 ```sh
pip install -r requirements.txt
  ```

---

## Build Package Instructions

### Build and Use the `.whl` File

1. **Build the `.whl` File in the root of this repo:**
  ```sh
  cd /path/to/Rhapso
  pip install setuptools wheel
  python setup.py sdist bdist_wheel
  ```
  The `.whl` file will appear in the `dist` directory. Do not rename it to ensure compatibility (e.g., `rhapso-0.1-py3-none-any.whl`).

2. **Install and Verify:**
  ```sh
  pip install dist/rhapso-0.1-py3-none-any.whl
  pip show rhapso
  ```

3. **Run Rhapso CLI:**
  ```sh
  Rhapso -h
  ```
  **Or import the package for scriping use:**
  ```python
  import Rhapso
  from Rhapso.solver.solver import Solver
  ```

---

## Usage:

### Python Pipeline Guide: Rhapso/pipelines/python_pipeline.py

### Overview
This pipeline provides a local execution environment for Rhapso, utilizing its core components to demonstrate the system's capabilities with sample data. It is designed to showcase how flexible and adaptable Rhapso is, allowing users to customize and interchange components to fit various use cases.

### Getting Started
1.	Location: The pipeline script is located at Rhapso/pipelines/python_pipeline.py.
2.	Purpose: Ideal for users new to Rhapso or those looking to explore its functionalities by stepping through the process with provided [sample data](#sample-data).
3.	Customization: Discover how to modify and tailor Rhapso to meet specific needs by experimenting with the components within this pipeline.

### Configuration
- Sample Data: To get started with the sample data, visit [sample data](#sample-data).
- Parameters: The pipeline requires setting parameters that vary based on your dataset size. These parameters are crucial for optimizing the execution of your pipeline.
-	Optimization: For detailed guidance on setting up optimization parameters, check out [Run Parameter Configurations](#run-parameter-configurations).

### Running the Job
1. Follow [Setup Instructions](#setup-instructions).
2. Navigate your terminal to the root folder of Rhapso. In your terminal, run: python Rhapso/pipelines/python_pipeline.py.

### Monitoring
Follow the steps in the pipeline script to understand the sequence and integration of Rhapso components. Each step is an opportunity to tweak and learn about the system’s flexibility in real-time applications.
<br>

### Spark ETL Pipeline Guide: Rhapso/pipelines/spark_etl_pipeline.py

### Overview
This pipeline enables the execution of Rhapso on production data using AWS Glue's Spark ETL capabilities, which is ideal for processing large-scale datasets, specifically when dealing with OME image data in terabytes or larger.

### Prerequisites
1.	AWS Account: Ensure you have an active AWS account. Sign up or log in here.
2.	Navigate to AWS Glue: Access AWS Glue from your AWS Management Console. Find it under "Services" or use the search bar.

### Setup
1.	Access ETL Jobs:
-	In AWS Glue, select "ETL Jobs" from the left sidebar.
-	Click on "Add Job" to start a new ETL job setup.
2.	Configure the Job:
-	Choose "Spark" as the ETL engine and select "Start Fresh" in the script editor.
-	In the script editor, paste the contents of Rhapso/pipelines/spark_etl_pipeline.py.
3.	Import Rhapso Library:
-	Navigate to "Job Details", then scroll to "Advanced Properties".
-	Under "Job Parameters", add --additional-python-modules as the key.
-	For the value, input the full S3 path to the .whl file containing the Rhapso project.
-	To create .whl file, navigate to [Build Package](#build-package)

### Running the Job
1.	Adjust Job Settings for Your Data:
-	Review guidelines on dataset sizes and optimal worker types [Run Parameter Configurations](#run-parameter-configurations) to ensure the Glue engine is configured correctly for your data.
2.	Save and Run:
-	Save your configurations and initiate the job by clicking "Run".
-	Monitor the job's progress in the "Runs" tab.

### Monitoring
Watch the execution in real-time and make any necessary adjustments based on the job performance and outputs.

---

## Components Walkthrough

This guide offers a high-level overview of Rhapso components, explaining each component's role in the process. It’s designed for users who want to understand or modify Rhapso’s process.

### Interest Point Detection
 Interest Point Detection involves detecting interest points by converting XML metadata into DataFrames, generating transformation matrices, detecting overlaps, loading and preprocessing image data,refining detected points, and saving the refined interest points for matching.

For more in depth information, checkout the [Detection ReadMe](./Rhapso/detection/readme.md) and the [detailed walkthrough on our wiki](https://github.com/AllenNeuralDynamics/Rhapso/wiki/1.-Detection).

### Interest Point Matching

Interest Point Matching involves loading and filtering interest points, organizing views, setting up pairwise matching, applying the RANSAC algorithm, refining matches, and compiling and storing results for Solver.

For more in depth information, checkout the [Matching ReadMe](./Rhapso/matching/readme.md) and the [detailed walkthrough on our wiki](https://github.com/AllenNeuralDynamics/Rhapso/wiki/2.-Matching).

### Solver

  Solver involves setting up models and tiles, aligning tiles using transformation models, performing optimization for consistency in preparation for Fusion.

For more in depth information, checkout the [Solver ReadMe](./Rhapso/solver/readme.md) and the [detailed walkthrough on our wiki](https://github.com/AllenNeuralDynamics/Rhapso/wiki/3.-Solver).

### Cloud Fusion
To Do

For more information about fusion, check out the [detailed walkthrough on our wiki](https://github.com/AllenNeuralDynamics/Rhapso/wiki/4.-Fusion).

---

## Frequently Asked Questions
