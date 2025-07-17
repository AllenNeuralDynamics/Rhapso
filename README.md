# Rhapso

**Rhapso** is a modular Python toolkit for aligning and fusing large-scale microscopy datasets. 

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/AllenNeuralDynamics/Rhapso/ci.yml?branch=main)](https://github.com/AllenNeuralDynamics/Rhapso/actions)
[![Documentation](https://img.shields.io/badge/docs-wiki-blue)](https://github.com/AllenNeuralDynamics/Rhapso/wiki)
[![Issues](https://img.shields.io/github/issues/AllenNeuralDynamics/Rhapso)](https://github.com/AllenNeuralDynamics/Rhapso/issues)

<!-- ## Example Usage Media Content Coming Soon....
-- -->

<br>

## Connect With Us
<!-- UPDATE THIS WHEN OPEN SOURCED -->
  [Allen Institute Internal Coms](https://teams.microsoft.com/l/channel/19%3AIv_3CdryfboX3E0g1BbzCi3Y8KRgNCdAv3idJ9epV-k1%40thread.tacv2/Project%20Rhapso-Shared?groupId=87b91b48-bb2a-4a00-bc59-5245043a0708&tenantId=32669cd6-737f-4b39-8bdd-d6951120d3fc&ngc=true&allowXTenantAccess=true)

<br>

## Table of Contents
- [Summary](#summary)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Build Package](#build-package)
  - [Using the Built `.whl` File](#using-the-built-whl-file)
- [Componenents Walkthrough](#components-walkthrough)
- [FAQ](#frequently-asked-questions)

---

<br>

## Summary
Rhapso is a tool that aligns and stitches tiled microscopy images into a single cohesive volume in four steps:
- **Interest Point Detection** – Identify distinctive features in each tile and record their coordinates.
- **Interest Point Matching** – Compare these features across tiles to find matching points (first rigid, then affine).
- **Solver** – Calculate transformations that align matched points, producing matrices that describe how each tile should be adjusted.
- **Fusion** – Use the computed alignment matrices to place and stitch all tiles into a coherent image.

Rhapso natively supports OME-TIFF and Zarr formats, and additional tile-based formats can be integrated by implementing a custom data loader. It also supports petabyte-scale datasets for distributed processing with Ray. 

To customize Rhapso (run environment, data loading method, really anything) create a custom pipeline script and plug in your custom components. Refer to our  [Wiki Page](https://github.com/AllenNeuralDynamics/Rhapso/wiki#example-pipelines) for guidance on adapting Rhapso configurations to your needs.

Rhapso is developed in collaboration with the Allen Institute for Neural Dynamics (AIND), initially supporting AIND’s ExaSPIM pipeline and eventually broadly benefiting microscopy research.

<br>

### Environments
This package is designed to be environment agnostic, but we have configured a few data loaders to run in these environments:
- Local (Dask, Ray)
- Cloud (AWS, Ray)
> [!TIP]
> Detailed instructions on how to run a sample end-to-end pipeline for each environment can be found on the [Wiki Page](https://github.com/AllenNeuralDynamics/Rhapso/wiki#example-pipelines).
Sample pipeline instructions are provided for pre-made templates, but if you want to create your own template, you can do so based on the sample files available inside the `Rhapso/pipelines` folder.

---

<br>

## Getting Started

### Clone Rhapso

 ```sh
git clone https://github.com/AllenNeuralDynamics/Rhapso.git
  ```
### Clone Rhapso Wiki
 ```sh
git clone https://github.com/AllenNeuralDynamics/Rhapso.wiki.git
  ```
### Download and Install Dependencies
 ```sh
pip install -r requirements.txt
  ```

---

<br>

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
1. Follow [Getting Started](#getting-started).
2. Navigate your terminal to the root folder of Rhapso. In your terminal, run: python Rhapso/pipelines/python_pipeline.py.

### Monitoring
Follow the steps in the pipeline script to understand the sequence and integration of Rhapso components. Each step is an opportunity to tweak and learn about the system’s flexibility in real-time applications.

---

<br>

### Ray Pipeline Guide: Rhapso/pipelines/detection/ray_pipeline.py

### Overview
This pipeline enables the execution of Rhapso on production data using Ray's distribted capabilities, which is ideal for processing large-scale datasets.

### Setup
1.	AWS Account: Ensure you have an active AWS account. Sign up or log in here.
2.	Create a PEM auth file and save locally
3.	Update ray_config.py file to point to your PEM path
     ```
     ssh_private_key:
     ```
4.	Navigate to project root and create .whl file (instructions at bottom of ray_config.py)
     ```
     python setup.py bdist_wheel 
     ```
5.	Save .whl file to desired location in S3
6.	Point to the .whl file in setup commands in ray_config.py
     ```
     - aws s3 cp s3://rhapso-whl-v2/Rhapso-0.1.0-py3-none-any.whl /tmp/Rhapso-0.1.0-py3-none-any.whl 
     ```
7.	Navigate to the root folder of ray_config.py
8.	Run ray up ray_cluster.yml --no-config-cache
    ```
    ray up ray_cluster.yml --no-config-cache
    ```

### Access Ray Dashboard
1.	Find public IP of head node.
2.	Replace the ip address and PEM file location to ssh into head node
     ```
    ssh -i /Users/seanfite/Desktop/AllenInstitute/Rhapso/Auth/AWS/EC2-SSH-Key/rhapso-ssh.pem -L 8265:localhost:8265 ubuntu@34.219.189.35
    ```
4.	Go to dashboard
     ```
    http://localhost:8265
    ```

---

<br>

## Build Package

### Using the Built `.whl` File

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

<br>

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

<br>

## Frequently Asked Questions
