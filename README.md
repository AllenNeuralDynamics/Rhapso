# Rhapso

This is the official code base for **Rhapso**, a modular Python toolkit for the alignment and stitching of large-scale microscopy datasets. 

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Documentation](https://img.shields.io/badge/docs-wiki-blue)](https://github.com/AllenNeuralDynamics/Rhapso/wiki)

<!-- ## Example Usage Media Content Coming Soon....
-- -->

<br>

## Table of Contents
- [Summary](#summary)
- [Contact](#contact)
- [Supported Features](#supported-features)
- [Performance](#performance)
- [Layout](#layout)
- [Installation](#installation)
- [How To Start](#how-to-start)
- [Try Rhapso on Sample Data](#try-rhapso-on-sample-data)
- [Ray](#ray)
- [Run Locally w/ Ray](#run-locally-with-ray)
- [Run on AWS Cluster w/ Ray](#run-on-aws-cluster-with-ray)
- [Access Ray Dashboard](#access-ray-dashboard)
- [Parameters](#parameters)
- [Tuning Guide](#tuning-guide)
- [Build Package](#build-package)
  - [Using the Built `.whl` File](#using-the-built-whl-file)

---

<br>

**Update 1/12/26** 
--------
Rhapso is still loading... and while we wrap up development, a couple things to know if you are outside the Allen Institute: 
   - This process requires a very specific XML structure to work.
   - Fusion/Mutliscale is included but still under testing and development

<br>

## Summary
Rhapso is a set of Python components used to register, align, and stitch large-scale, overlapping, tile-based, multiscale microscopy datasets. Its stateless components can run on a single machine or scale out across cloud-based clusters. 

Rhapso is published on PyPI.

Rhapso was developed by the Allen Institute for Neural Dynamics.

<br>

## Contact
Questions or want to contribute? Please open an issue..

<br>

## Supported Features
- **Interest Point Detection** - DOG based feature detection
- **Interest Point Matching** - Descriptor based RANSAC to match feature points
- **Global Optimization** - Align matched features between tile pairs globally
- **Validation and Visualization Tools** - Validate component specific results for the best output
- **ZARR** - Zarr data as input
- **TIFF** - TIFF data as input
- **AWS** - AWS S3 based input/output and Ray based EC2 instances
- **Scale** - Tested on 200 TB of data without downsampling

---

<br>


## Layout

```
Rhapso/
└── Rhapso/
    ├── data_prep/                          # Custom data loaders
    ├── detection/
    ├── evaluation/
    ├── fusion/
    ├── image_split/
    ├── matching/
    ├── pipelines/
    │   └── ray/
    │       ├── aws/
    │       │   ├── config/                 # Cluster templates (edit for your account)
    │       │   └── alignment_pipeline.py   # AWS Ray pipeline entry point
    │       ├── local/
    │       │   └── alignment_pipeline.py   # Local Ray pipeline entry point
    │       ├── param/                      # Run parameter files (customize per run)
    │       ├── interest_point_detection.py # Detection pipeline script
    │       ├── interest_point_matching.py  # Matching pipeline script
    │       └── solver.py                   # Global solver script
    ├── solver/
    └── visualization/                      # Validation tools
```

---

<br>


## Installation

### Option 1: Install from PyPI (recommended)

```bash
# create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate
# or: conda create -n rhapso python=3.10 && conda activate rhapso

# install Rhapso from PyPI
pip install Rhapso
```

### Option 2: Install from GitHub (developers)

 ```sh
# clone the repo
git clone https://github.com/AllenNeuralDynamics/Rhapso.git

# create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate
# or: conda create -n rhapso python=3.11 && conda activate rhapso

# install deps
pip install -r requirements.txt
```
---

<br>

## How to Start

Rhapso is driven by **pipeline scripts**.

- Each pipeline script has at minimum an associated **param file** (e.g. in `Rhapso/pipelines/ray/param/`).
- If you are running on a cluster, you’ll also have a **Ray cluster config** (e.g. in `Rhapso/pipelines/ray/aws/config/`).

A good way to get started:

1. **Pick a template pipeline script**  
   For example:
   - `Rhapso/pipelines/ray/local/alignment_pipeline.py` (local)
   - `Rhapso/pipelines/ray/aws/alignment_pipeline.py` (AWS/Ray cluster)

3. **Point it to your param file**  
   Update the `with open("...param.yml")` line so it reads your own parameter YAML.
   - [Run Locally w/ Ray](#run-locally-with-ray)

5. **(Optional) Point it to your cluster config**  
   If you’re using AWS/Ray, update the cluster config path.
   - [Run on AWS Cluster w/ Ray](#run-on-aws-cluster-with-ray)

5. **Edit the params to match your dataset**  
   Paths, downsampling, thresholds, matching/solver settings, etc.

6. **Run the pipeline**  
   The pipeline script will call the Rhapso components (detection, matching, solver, fusion) in the order defined in the script using the parameters you configured.

---

<br>

## Try Rhapso on Sample Data

The quickest way to get familiar with Rhapso is to run it on a real dataset. We have a small (10GB) Z1 example hosted in a public S3 bucket, so you can access it without special permissions. It’s a good starting point to copy and adapt for your own alignment workflows.

XML (input)
- s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_tile_alignment/single_channel_xmls/channel_488.xml

Image prefix (referenced by the XML)
- s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_radial_correction/

<br>

**Note:** Occasionally we clean up our aind-open-data bucket. If you find this dataset does not exist, please create an issue and we will replace it.

---

<br>

## High Level Approach to Registration, Alignment, and Fusion

This process has a lot of knobs and variations, and when used correctly, can work for a broad range of datasets.

**First, figure out what type of alignment you need.**  
- Are there translations to shift to?  
- If so, you’ll likely want to start with a rigid alignment.

Once you’ve run the rigid step, how does your data look?  
- Did the required translations shrink to an acceptable level?  
- If not, try again with new parameters, keeping the questions above in mind.

At this point, the translational part of your alignment should be in good shape. Now ask: **are transformations needed?** If so, you likely need an affine alignment next.

Your dataset should be correctly aligned at this point. If not, there are a number of reasons why, and we have listed some common recurrences and will keep this up to date.

There is a special case in some datasets where the z-stack is very large. In this case, you can use the split-dataset utility, which splits each tile into chunks. Then you can run split-affine alignment, allowing for more precise transformations without such imposing global rails.

**Common Causes of Poor Alignment**
- Not enough quality matches (adjust sigma threshold until you do)
- Data is not consistent looking (we take a global approach to params)
- Large translations needed (extend search radius)
- Translations that extend beyond overlapping span (increase overlap)

---

<br>

## Performance

**Interest Point Detection Performance Example (130TB Zarr dataset)**

| Environment           | Resources            | Avg runtime |
|:----------------------|:---------------------|:-----------:|
| Local single machine  | 10 CPU,  10 GB RAM   | ~120 min    |
| AWS Ray cluster       | 560 CPU, 4.4 TB RAM  | ~30 min     |

<br>
*Actual times vary by pipeline components, dataset size, tiling, and parameter choices.*

---

<br>

## Ray

**Ray** is a Python framework for parallel and distributed computing. It lets you run regular Python functions in parallel on a single machine **or** scale them out to a cluster (e.g., AWS) with minimal code changes. In Rhapso, we use Ray to process large scale datasets.

- Convert a function into a distributed task with `@ray.remote`
- Control scheduling with resource hints (CPUs, memory)

<br>
  
> [!TIP]
> Ray schedules **greedily** by default and each task reserves **1 CPU**, so if you fire many tasks, Ray will try to run as many as your machine advertises—often too much for a laptop. Throttle concurrency explicitly so you don’t overload your system. Use your machine's activity monitor to track this or the Ray dashboard to monitor this on your cluster:
>
> - **Cap by CPUs**:
>   ```python
>   @ray.remote(num_cpus=3)   # Ray will schedule each time 3 cpus are available
>   ```
> - **Cap by Memory and CPU** if Tasks are RAM-Heavy (bytes):
>   ```python
>   @ray.remote(num_cpus=2, memory=4 * 1024**3)  # 4 GiB and 2 CPU per task>
>   ```
> - **No Cap** on Resources:
>   ```python
>   @ray.remote             
>   ```
> - **Good Local Default:**
>   ```python
>   @ray.remote(num_cpus=2)
>   ```

---

<br>


## Run Locally with Ray

### 1. Edit or create param file (templates in codebase)
```python
Rhapso/Rhapso/pipelines/param/
```

### 2. Update alignment pipeline script to point to param file
```python
with open("Rhapso/pipelines/ray/param/your_param_file.yml", "r") as file:
    config = yaml.safe_load(file)
```

### 3. Run local alignment pipeline script
```python
python Rhapso/pipelines/ray/local/alignment_pipeline.py

```

---

<br>


## Run on AWS Cluster with Ray

### 1. Edit/create param file (templates in codebase)
```python
Rhapso/pipelines/ray/param/
```

### 2. Update alignment pipeline script to point to param file
```python
with open("Rhapso/pipelines/ray/param/your_param_file.yml", "r") as file:
    config = yaml.safe_load(file)
```

### 3. Edit/create config file (templates in codebase)
```python
Rhapso/pipelines/ray/aws/config/
```

### 5. Update alignment pipeline script to point to config file
```python
unified_yml = "your_cluster_config_file_name.yml"
```

### 7. Run AWS alignment pipeline script
```python
python Rhapso/pipelines/ray/aws/alignment_pipeline.py
```

> [!TIP]
> - The pipeline script is set to always spin the cluster down, it is a good practice to double check in AWS.
> - If you experience a sticky cache on run params, you may have forgotten to spin your old cluster down.

<br>

## Access Ray Dashboard

**This is a great place to tune your cluster's performance.**
1.	Find public IP of head node.
2.	Replace the ip address and PEM file location to ssh into head node.
     ```
    ssh -i /You/path/to/ssh/key.pem -L port:localhost:port ubuntu@public.ip.address
    ```
4.	Go to dashboard.
     ```
    http://localhost:8265
    ```

---

<br>

## Parameters

### Detection
```
| Parameter          | Feature / step         | What it does                                                                                  | Typical range\*                   |
| :----------------- | :--------------------- | :-------------------------------------------------------------------------------------------- | :-------------------------------- |
| `dsxy`             | Downsampling (XY)      | Reduces XY resolution before detection; speeds up & denoises, but raises minimum feature size | 16                                |
| `dsz`              | Downsampling (Z)       | Reduces Z resolution; often lower than XY due to anisotropy                                   | 16                                |
| `min_intensity`    | Normalization          | Lower bound for intensity normalization prior to DoG                                          | 1                                 |
| `max_intensity`    | Normalization          | Upper bound for intensity normalization prior to DoG                                          | 5                                 |
| `sigma`            | DoG blur               | Gaussian blur scale (sets feature size); higher = smoother, fewer peaks                       | 1.5 - 2.5                         |
| `threshold`        | Peak detection (DoG)   | Peak threshold (initial min peak ≈ `threshold / 3`); higher = fewer, stronger peaks           | 0.0008 - .05                      |
| `median_filter`    | Pre-filter (XY)        | Median filter size to suppress speckle/isolated noise before DoG                              | 1-10                              |
| `combine_distance` | Post-merge (DoG peaks) | Merge radius (voxels) to de-duplicate nearby detections                                       | 0.5                               |
| `chunks_per_bound` | Tiling/parallelism     | Sub-partitions per tile/bound; higher improves parallelism but adds overhead                  | 12-18                             |
| `max_spots`        | Post-cap               | Maximum detections per bound to prevent domination by dense regions                           | 8,0000 - 10,000                   |
```
<br>

### Matching
```
# Candidate Selection
| Parameter                      | Feature / step      | What it does                                                      | Typical range  |
| :----------------------------- | :------------------ | :---------------------------------------------------------------- | :------------- |
| `num_neighbors`                | Candidate search    | Number of nearest neighbors to consider per point                 | 3              |
| `redundancy`                   | Candidate search    | Extra neighbors added for robustness beyond `num_neighbors`       | 0 - 1          |
| `significance`                 | Ratio test          | Strictness of descriptor ratio test; larger = stricter acceptance | 3              |
| `search_radius`                | Spatial gating      | Max spatial distance for candidate matches (in downsampled units) | 100 - 300      |
| `num_required_neighbors`       | Candidate filtering | Minimum neighbors required to keep a candidate point              | 3              |

# Ransac
| Parameter                     | Feature / step       | What it does                                                      | Typical range  |
| :---------------------------- | :------------------- | :---------------------------------------------------------------- | :------------- |
| `model_min_matches`           | RANSAC               | Minimum correspondences to estimate a rigid transform             | 18 – 32        |
| `inlier_factor`               | RANSAC               | Inlier tolerance scaling; larger = looser inlier threshold        | 30 – 100       |
| `lambda_value`                | RANSAC               | Regularization strength during model fitting                      | 0.1 – 0.05     |
| `num_iterations`              | RANSAC               | Number of RANSAC trials; higher = more robust, slower             | 10,0000        |
| `regularization_weight`       | RANSAC               | Weight applied to the regularization term                         | 1.0            |

```
<br>

### Solver
```
| Parameter            | Feature / step | What it does                                                       | Typical range       |
| :------------------- | :------------- | :----------------------------------------------------------------- | :------------------ |
| `relative_threshold` | Graph pruning  | Reject edges with residuals above dataset-relative cutoff          | 3.5                 |
| `absolute_threshold` | Graph pruning  | Reject edges above an absolute error bound (detection-space units) | 7.0                 |
| `min_matches`        | Graph pruning  | Minimum matches required to retain an edge between tiles           | 3                   |
| `damp`               | Optimization   | Damping for iterative solver; higher can stabilize tough cases     | 1.0                 |
| `max_iterations`     | Optimization   | Upper bound on solver iterations                                   | 10,0000             |
| `max_allowed_error`  | Optimization   | Overall error cap; `inf` disables hard stop by error               | `inf`               |
| `max_plateauwidth`   | Early stopping | Stagnation window before stopping on no improvement                | 200                 |

```

---

<br>

## Tuning Guide

- **Start with Detection.** The quality and density of interest points strongly determine alignment outcomes.

- **Target Counts (exaSPIM):** ~25–35k points per tile in dense regions; ~10k for sparser tiles. Going much higher usually increases runtime without meaningful accuracy gains.

- **Inspect Early.** After detection, run the visualization script and verify that peaks form **clustered shapes/lines** with a **good spatial spread**—a good sign for robust rigid matches.

- **Rigid → Affine Dependency.** Weak rigid matches produce poor rigid transforms, which then degrade affine matching (points don’t land close enough). If tiles fail to align:
  - Check **match counts** for the problem tile and its neighbors.
  - Adjust high-impact detection knobs—`sigma`, `threshold`, and `median_filter`—within sensible ranges.
  - Revisit `max_spots` and `combine_distance` to balance density vs. duplicate detections.

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

---

<br>
<br>
<br>
