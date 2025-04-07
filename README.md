# Rhapso

Rhapso is a Python-based tool designed to robustly and efficiently align and fuse large-scale microscopy datasets. The tool is built with flexibility, scalability, and modularity in mind, making it suitable for cloud-native deployments, as well as on-prem or cluster-based executions. It aims to improve existing imaging pipelines by incorporating performance optimizations, enhanced robustness, and automation through machine learning.

Rhapso is being developed as part of the Allen Institute for Neurotechnology (AIND) and will be published as an open-source software component in the OCTO SDK for image processing. Initially, it will benefit AIND's ExaSPIM pipeline and the broader scientific community in their microscopy research.

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)

## Table of Contents
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Componenents Walkthrough](#components-walkthrough)
- [Sample Data](#sample-data)
- [Run Parameter Configurations](#run-parameter-configurations)
- [Command Line Usage](#command-line-usage)
- [Setup Instructions](#setup-instructions)
- [Build Package](#build-package)
  - [Using the Built `.whl` File](#using-the-built-whl-file)
- [Run Tests](#run-tests)
- [Environments](#environments)
- [Use Cases](#use-cases)
- [Cloud Deployment Plan](#cloud-deployment-plan)
- [To Do](#to-do)

---

## Repository Structure

```
rhapso/
│
├── Rhapso/                      # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # Command-line interface entry point
│   ├── pipeline.py              # Pipeline demo/example file
│
│   ├── detection/               # Detection algorithms
│   │   ├── __init__.py
│   │   ├── save_interest_points.py  # Save interest points
│   │   ├── overlap_detection.py     # Overlap detection
│   │   ├── interest_points.py       # Interest point detection CLI
│   │   ├── interest_point_detection.py  # Interest point detection logic
│   │   ├── filtering_and_optimization.py  # Filtering and optimization
│   │   └── advanced_refinement.py    # Advanced refinement
│
│   ├── fusion/                  # Fusion methods
│   │   ├── __init__.py
│   │   └── affine_fusion.py     # Affine fusion
│
│   ├── matching/                # Matching algorithms
│   │   ├── __init__.py
│   │   └── feature_matching.py  # Feature-based matching
│
│   ├── solving/                 # Solvers for optimization
│   │   ├── __init__.py
│   │   └── solver.py  

│   ├── data_preparation/        # Data preparation methods
│   │   ├── __init__.py
│   │   ├── xml_to_dataframe.py  # XML to DataFrame conversion
│   │   └── dataframe_to_xml.py  # DataFrame to XML conversion
│
├── tests/                       # Unit tests for each module
│   ├── test_detection.py
│   ├── test_data_preparation.py
│   └── __init__.py
│
├── setup.py                     # Package installation
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies
```

---


## Usage

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
1.	**XML to DataFrame** -	Converts XML metadata into structured DataFrames to facilitate data manipulation.
2.	**View Transform Models** -	Generates transformation matrices from XML data to align multiple views in a dataset.
3.	**Overlap Detection** -	Identifies overlapping areas between different views using the transformation matrices.
4.	**Load Image Data** -	Loads and preprocesses image data based on detected overlaps, preparing it for feature detection.
5.	**Difference of Gaussian** - Applies the Difference of Gaussian (DoG) method to identify potential interest points in the image data.
6.	**Advanced Refinement** -	Refines detected points using a KD-tree structure to ensure accuracy and relevance of features.
7.	**Save Interest Points** - Saves the refined interest points and associated metadata for further analysis or usage.

### Interest Point Matching
1.	**XML Parsing** -	Extracts necessary metadata such as view IDs and setup information from an XML file, crucial for correlating different datasets.
2.	**Data Retrieval** - Fetches data from specified sources (local or cloud storage) based on the XML configuration, ensuring that all relevant image data is accessible for processing.
3.	**Interest Points Loading** -	Loads interest points data, which contains critical features extracted from images. This step is essential for subsequent matching procedures.
4.	**Interest Points Filtering** -	Filters out irrelevant or less significant points based on predefined criteria, refining the dataset for more accurate matching.
5.	**View Grouping** -	Organizes views into logical groups, facilitating efficient and systematic pairing for the matching process.
6.	**Pairwise Matching Setup** -	Prepares and configures the conditions and parameters for pairwise matching between the grouped views.
7.	**RANSAC for Matching** -	Applies the RANSAC algorithm to find the best match between pairs, using geometric constraints to validate the correspondences.
8.	**Match Refinement** - Refines the matches to ensure high accuracy, discarding outliers and confirming valid correspondences based on robust statistical methods.
9.	**Results Compilation and Storage** -	Aggregates all matching results and stores them in a designated format and location for further analysis or use in downstream processes.

### Solver
1.	**XML to DataFrame** - Converts XML metadata into structured DataFrames to facilitate data manipulation and subsequent operations.
2.	**View Transform Models** -	Generates affine transformation matrices from the DataFrames, essential for aligning multiple views in a coherent manner.
3.	**Data Preparation** - Prepares and organizes data retrieved from different sources, setting the stage for effective model generation and tile setup.
4.	**Model and Tile Setup** - Creates models and configures tiles based on the prepared data and the transformation matrices, crucial for the optimization process.
5.	**Align Tiles** -	Applies transformation models to tiles, aligning them according to the specified parameters and conditions.
6.	**Global Optimization** -	Performs a comprehensive optimization over all tiles to refine the alignment based on a global perspective, ensuring consistency and accuracy across the dataset.
7.	**Save Results** - Saves the optimized results back to XML, documenting the new affine transformations for each view, thereby finalizing the process.

### Cloud Fusion
To Do

---

## Sample Data
To Do - add sample data sets for users to download

---

## Run Parameter Configurations
To Do - add configurations

---

## Command Line Usage

After installing Rhapso, you can use the following commands:

### View General Help

```bash
Rhapso -h
```

### View Subcommand-Specific Help

- **Detect Help**

```bash
Rhapso detect -h
```

- **Match Help**

```bash
Rhapso match -h
```

- **Fuse Help**

```bash
Rhapso fuse -h
```

- **Solve Help**

```bash
Rhapso solve -h
```

---

## Example CLI Commands

1. **Detect Interest Points**
```bash
Rhapso detect --sigma 1.8 --threshold 0.05 --medianFilter 10
```

2. **Match Features**
```bash
Rhapso match --x debug --l debug --method ICP --tiffPath "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/IP_TIFF_XML (after detection)/spim_TL18_Angle0.tif"
```

3. **Affine Fusion**
```bash
Rhapso fuse --scale 2.0 --output ./output/fused.tif --blend
```

4. **Solve Transformations**
```bash
Rhapso solve --method Affine --iterations 50 --tolerance 0.0005
```

---

## Setup Instructions

### Clone Git Repository
```sh
git clone https://github.com/AllenNeuralDynamics/Rhapso.git
cd Rhapso
```

### Setup Python Virtual Environment
```sh
python -m venv virtenv
# Windows
virtenv\Scripts\activate
# Mac/Linux
source virtenv/bin/activate
```

### Download and Install Dependencies
Before installing the Rhapso package, ensure all necessary dependencies are installed:
```sh
pip install -r requirements.txt
```

### Install Rhapso Package Locally
```sh
pip install -e .
```

### Check Installed Rhapso Version
```sh
pip show rhapso
```

### Run Rhapso Package
```sh
# Using command line
Rhapso alice

# Using Python module
python -m Rhapso

# Using Python script
import Rhapso
print(Rhapso.__version__)
```

### Uninstall Rhapso Package
```sh
pip uninstall rhapso
```

---

## Build Package

To build the Rhapso package as a `.whl` file, follow these steps:

1. **Navigate to the project directory:**
   ```sh
   cd /path/to/Rhapso
   ```

2. **Ensure you have the required build tools installed:**
   ```sh
   pip install setuptools wheel
   ```

3. **Build the package:**
   ```sh
   python setup.py sdist bdist_wheel
   ```

4. **The built `.whl` file will be located in the `dist` directory.**

### Using the Built `.whl` File

1. **Navigate to the `dist` directory:**
   ```sh
   cd dist
   ```

2. **Install the `.whl` file:**
   ```sh
   pip install rhapso-0.1-py3-none-any.whl
   ```

3. **Verify the installation:**
   ```sh
   pip show rhapso
   ```

4. **Run the Rhapso CLI:**
   ```sh
   Rhapso -h
   ```

---

## Run Tests

To run the tests, use the following command:
```sh
python -m unittest discover
```

---

## Environments

- **Dev**
- **Prod**

---

## Use Cases

### Fully Local

```sh
pip install Rhapso
Rhapso detect --i '../../dataset.zarr' --o '../dataset_with_ip.xml'
```

### Fully in Cloud

Install Rhapso package and call it either via CLI, Python module, or as a Python library in scripts.

### Cloud/Local Hybrid

```sh
pip install Rhapso
aws configure
Rhapso detect --i 's3://data.zarr' --o 's3://output/my_dataset.xml'
```

---

## Cloud Deployment Plan

### Pushing with GitHub Actions Workflow to S3 Bucket

1. **Create an S3 Bucket:**
   - Go to the AWS Management Console.
   - Navigate to S3 and create a new bucket (e.g., `rhapso-deployments`).

2. **Set Up IAM Role and Policy:**
   - Create an IAM role with permissions to access the S3 bucket.
   - Attach the following policy to the role:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Action": [
             "s3:PutObject",
             "s3:GetObject",
             "s3:ListBucket"
           ],
           "Resource": [
             "arn:aws:s3:::rhapso-deployments",
             "arn:aws:s3:::rhapso-deployments/*"
           ]
         }
       ]
     }
     ```

3. **Configure GitHub Secrets:**
   - Go to your GitHub repository settings.
   - Add the following secrets:
     - `AWS_ACCESS_KEY_ID`
     - `AWS_SECRET_ACCESS_KEY`

4. **GitHub Actions Workflow:**
   - Create a GitHub Actions workflow file (e.g., `.github/workflows/deploy.yml`) with the following content:
     ```yaml
     name: Deploy to S3

     on:
       push:
         branches:
           - main

     jobs:
       deploy:
         runs-on: ubuntu-latest
         steps:
           - name: Checkout code
             uses: actions/checkout@v4

           - name: Configure AWS credentials
             uses: aws-actions/configure-aws-credentials@v1
             with:
               aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
               aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
               aws-region: us-west-2

           - name: Sync files to S3
             run: |
               aws s3 sync . s3://rhapso-deployments --exclude ".git/*" --delete
     ```

### Setting Up Permissions with IAM

1. **Create IAM User:**
   - Go to the AWS Management Console.
   - Navigate to IAM and create a new user (e.g., `rhapso-deployer`).
   - Attach the following policy to the user:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Action": [
             "s3:PutObject",
             "s3:GetObject",
             "s3:ListBucket"
           ],
           "Resource": [
             "arn:aws:s3:::rhapso-deployments",
             "arn:aws:s3:::rhapso-deployments/*"
           ]
         }
       ]
     }
     ```

2. **Generate Access Keys:**
   - Generate access keys for the IAM user.
   - Store the access keys securely and add them as GitHub secrets.

3. **Revoking Credentials:**
   - To revoke credentials, go to the IAM user in the AWS Management Console.
   - Delete the access keys or deactivate the user.
   - Generate new access keys and update the GitHub secrets.

### Cloud Deployment Image

![Cloud Deployment](docs/deployment.png)

---

### To Do:
- Setup and add Tests 
- Improve `setup.py` to include more metadata details about this package.


### Example Commands:
```sh

# Example script run for import method
python example.py 

# Run locally with local xml file
Rhapso --xmlToDataframe /mnt/c/Users/marti/Documents/Allen/repos/Rhapso-Sample-Data/IP_TIFF_XML/dataset.xml

# Run locally with cloud s3 xml file (must run aws configure first, and give iam access with correct s3 permission) 
Rhapso --xmlToDataframe s3://rhapso-dev/rhapso-sample-data/dataset.xml

# Run overlap detection locally with a local xml file
Rhapso --xmlToDataframe ../../demo/dataset.xml --runOverlapDetection

# Run overlap detection locally with a aws s3 cloud xml file
Rhapso --xmlToDataframe s3://rhapso-dev/rhapso-sample-data/dataset.xml --runOverlapDetection
```
Run overlap detection in Python script:
```
import Rhapso

# Call the xmlToDataframe function
# myDataframe = Rhapso.xmlToDataframe("/mnt/c/Users/marti/Documents/Allen/repos/Rhapso-Sample-Data/IP_TIFF_XML/dataset.xml")
myDataframe = Rhapso.xmlToDataframe("s3://rhapso-dev/rhapso-sample-data/dataset.xml")
print('myDataframe = ', myDataframe)

# Call the runOverlapDetection function
overlapDetection = Rhapso.OverlapDetection()
output = overlapDetection.run(myDataframe)
print("Overlap Detection Output: ", output)
```
