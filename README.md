# Rhapso

This is the official code base for **Rhapso**, a modular Python toolkit for stitching (alignment and fusion) large-scale microscopy datasets. 

Now Supporting Zarr v3

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-wiki-blue)](https://github.com/AllenNeuralDynamics/Rhapso/wiki)

> Rhapso is published on PyPI and developed by the Allen Institute.

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

## Summary
Rhapso is a set of Python components used to register, align, and fuse large-scale images. Its stateless components can run on a single machine or scale out across cloud-based clusters. 

The core Rhapso components are not tied to any specific cloud service, infrastructure, image type, or imaging modality. The only requirement is 3D image data in NumPy arrays. The initial use case is large-scale, overlapping, tile-based, multiscale 3D microscopy datasets (OME Zarr) hosted in S3, but the same components can be adapted to many other large image workflows.

**Looking forward, we are developing an automated QC system for alignment.**

<br>

## Contact
Questions or want to contribute? Please open an issue..

<br>

## Supported Features
- **Interest Point Detection** - DOG based feature detection
- **Interest Point Matching** - Descriptor based RANSAC to match feature points
- **Global Optimization** - Align matched features between tile pairs globally
- **Affine Fusion** - Fuse tiles using generated alignments up to affine
- **Multiscale** - Add lower resolution scaling to dataset
- **Validation and Visualization Tools** - Validate component specific results for the best output
- **ZARR** - Zarr data as input
- **TIFF** - TIFF data as input
- **AWS** - AWS S3 based input/output and Ray based EC2 instances
- **Scale** - Tested on 130 TB of data without downsampling

---

<br>


## Layout

```
Rhapso/
└── rhapso/
    ├── data_prep/        # Data readers and XML/DataFrame preparation
    ├── detection/        # Difference-of-Gaussian interest point detection
    ├── matching/         # RANSAC-based interest point matching
    ├── solver/           # Global optimization and transform solving
    ├── affine_fusion/    # Affine fusion 
    ├── multiscale/       # Multiscale OME-Zarr pyramid generation
    ├── split_dataset/    # Dataset splitting utilities
    ├── evaluation/       # QC and visualization helpers
    ├── util/             # Miscellaneous XML/QC/Neuroglancer utilities
    └── pipelines/
        └── ray/
            ├── aws/      # AWS Ray cluster entry points and config templates
            ├── local/    # Local Ray entry points
            ├── param/    # Example/template YAML parameter files
            ├── interest_point_detection.py
            ├── interest_point_matching.py
            ├── solver.py
            ├── affine_fusion.py
            ├── multiscale.py
            └── split_dataset.py
```

---

<br>


## Installation

Rhapso requires Python 3.11 or newer. The examples below use Python 3.11.

### Option 1: Install from PyPI (recommended)

#### macOS and Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install Rhapso
```

#### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install Rhapso
```

#### Conda

```bash
conda create -n rhapso python=3.11
conda activate rhapso

python -m pip install Rhapso
```

### Option 2: Install from GitHub (developers)

```bash
git clone https://github.com/AllenNeuralDynamics/Rhapso.git
cd Rhapso

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, use the PowerShell environment creation and activation commands shown above, then run `python -m pip install -e .` from the repository root. The editable install includes the dependencies declared by Rhapso.

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
   - `Rhapso/pipelines/ray/local/fusion_pipeline.py` (local)
   - `Rhapso/pipelines/ray/aws/fusion_pipeline.py` (AWS/Ray cluster)

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

Use the [Parameters](#parameters) table and [Tuning Guide](#tuning-guide) if you need help picking params.

<br>

**Note:** Occasionally we clean up our aind-open-data bucket. If you find this dataset does not exist, please create an issue and we will replace it.

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
Rhapso/pipelines/ray/param/
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

There is no single set of parameters that will work well for every dataset. The optimal values depend on the characteristics of the data, the dataset size, and the accuracy required for alignment.

The ranges below are examples intended to explain the available parameters, what they control, and the values commonly used for some of our light-sheet datasets. They should be treated as starting points and adjusted based on the quality of the detected peaks, matches, and final alignment.

<br>

### Detection
```
| Parameter          | Feature / step         | What it does                                                            | Example range\*      |
| :----------------- | :--------------------- | :---------------------------------------------------------------------- | :------------------- |
| `dsxy`             | Downsampling (XY)      | Reduces XY resolution                                                   | 1, 2, 4, 8, or 16    |
| `dsz`              | Downsampling (Z)       | Reduces Z resolution                                                    | 1, 2, 4, 8, or 16    |
| `min_intensity`    | Normalization          | Lower bound for intensity normalization prior to DoG                    | 0                    |
| `max_intensity`    | Normalization          | Upper bound for intensity normalization prior to DoG                    | 100                  |
| `sigma`            | DoG blur               | Gaussian blur scale (sets feature size), higher = smoother              | 1.1 - 2.8            |
| `threshold`        | Peak detection (DoG)   | Peak gating threshold, higher = fewer points                            | 0.0001 - .9          |
| `median_filter`    | Pre-filter (XY)        | Median filter size to suppress speckle/isolated noise before DoG        | 8                    |
| `combine_distance` | Post-merge (DoG peaks) | Merge radius (voxels) to de-duplicate nearby detections                 | 1-16                 |
| `chunks_per_bound` | Tiling/parallelism     | Amount of chunks per overlap bound, to optimize mem usage and run time  | 1 - 50               |
| `max_spots`        | Post-cap               | Maximum detections per bound to prevent domination by dense regions     | 0 - 100,000          |
```
<br>

### Matching
```
| Parameter                 | Feature / step      | What it does                                                      | Example range  |
| :------------------------ | :------------------ | :---------------------------------------------------------------- | :------------- |
| `num_neighbors`           | Candidate search    | Number of nearest neighbors to consider per point                 | 3              |
| `redundancy`              | Candidate search    | Extra neighbors added for robustness beyond `num_neighbors`       | 0 - 1          |
| `significance`            | Ratio test          | Strictness of descriptor ratio test; larger = stricter acceptance | 3 - 5          |
| `search_radius`           | Spatial gating      | Max spatial distance for candidate matches (in downsampled units) | 100 - 600      |
| `num_required_neighbors`  | Candidate filtering | Minimum neighbors required to keep a candidate point              | 3              |
| `ransac_sample_size`      | RANSAC              | Minimum sample size                                               | 3 - 5          |
| `model_min_inliers`       | RANSAC              | Minimum correspondences to estimate a transform                   | 18 – 32        |
| `inlier_factor`           | RANSAC              | Inlier tolerance scaling; larger = looser inlier threshold        | 30 – 100       |
| `lambda_value`            | RANSAC              | Regularization strength during model fitting                      | 0.1 – 0.05     |
| `num_iterations`          | RANSAC              | Number of RANSAC trials; higher = more robust, slower             | 10,0000        |
| `regularization_weight`   | RANSAC              | Weight applied to the regularization term                         | .05 - 1.0      |

```
<br>

### Solver
```
| Parameter            | Feature / step | What it does                                                       | Example range       |
| :------------------- | :------------- | :----------------------------------------------------------------- | :------------------ |
| `relative_threshold` | Graph pruning  | Reject edges with residuals above dataset-relative cutoff          | 3.5                 |
| `absolute_threshold` | Graph pruning  | Reject edges above an absolute error bound (detection-space units) | 7.0                 |
| `max_cleanup_rounds` | Graph pruning  | Number of cleanup rounds                                           | 3 - 5               |
| `min_matches`        | Graph pruning  | Minimum matches required to retain an edge between tiles           | 3                   |
| `damp`               | Optimization   | Damping for iterative solver; higher can stabilize tough cases     | 1.0                 |
| `max_iterations`     | Optimization   | Upper bound on solver iterations                                   | 10,0000             |
| `max_allowed_error`  | Optimization   | Overall error cap; `inf` disables hard stop by error               | `inf`               |
| `max_plateauwidth`   | Early stopping | Stagnation window before stopping on no improvement                | 200                 |

```
<br>

### Split
```
| Parameter            | Feature / step | What it does                                       | Example range       |
| :------------------- | :------------- | :------------------------------------------------- | :------------------ |
| `point_density`      | Fake points    | Controls overlap point count                       | 1.0                 |
| `min_points`         | Fake points    | Min points per overlap                             | 20                  |
| `max_points`         | Fake points    | Max points per overlap                             | 800                 |
| `error`              | Fake points    | Adds random coordinate jitter                      | 0.5                 |
| `exclude_radius`     | Fake points    | Prevents points from being too close               | 200                 |
| `target_image_size`  | Grid split     | Desired split tile size                            | [5000, 5000, 3400]  |
| `target_overlap`     | Grid split     | Desire tile overlap size                           | [128, 128, 128]     |

```
<br>

### Fusion
```
| Parameter            | Feature / step | What it does                                 | Example range                             |
| :------------------- | :------------- | :--------------------------------------------| :---------------------------------------- |
| `block_size`         | Fusion opt     | Cell size per task xyz                       | 256, 256, 256                             |
| `intensity_range`    | Fusion config  | Range of intensity values                    | 0, 65535                                  |
| `block_scale`        | Fusion opt     | Scaling of cell size                         | 2, 2, 1                                   |
| `overlap_strategy`   | Edge handling  | Strategy for competing pixels                | avg_blend, lowest_view_wins, or max_blend |
| `output_zarr_version`| Zarr version   | Set which zarr version you want for output   | 2 or 3                                    |

```
<br>

### Multiscale
```
| Parameter               | Feature / step         | What it does                                            | Example range       |
| :---------------------- | :----------------------| :------------------------------------------------------ | :------------------ |
| `multiscale_chunk_size` | Optimization           | Output cell size                                        | 128, 128, 128       |
| `voxel_size`            | Data config            | Voxel size of data in zyx                               | 1.0, .748, .748     |
| `n_lvls`                | Output scale handling  | Num levels to multiscale including base level           | 7                   |
| `scale_factor`          | Entropy config         | Scaling factor per level for entropy                    | [2,2,2],...num lvls |
| `target_block_size_mb`  | Optimization           | Per worker block size                                   | 256                 |
| `base_level`            | Output scale config    | Existing base res level                                 | 0                   |

```

---

<br>

## Tuning Guide

The alignment workflow has three main components: **peak detection, matching, and solving**. These components are tuned together, and the matching and solver stages can be run progressively through **rigid, affine, and split-affine alignment**.

### 1. Peak Detection

* **Inspect Your Data:** Start by understanding what your data looks like and what type of registration it requires. Determine whether feature-based registration is appropriate and whether the misalignment can be corrected with a rigid transformation or requires affine deformation.

* **Tune Peak Detection:** Choose a downsampling level that balances dataset size, runtime, and the precision required for alignment. Sigma should correspond to the expected size of the features you want to detect, while the threshold controls how much noise is allowed through. Interest-point quality and spatial coverage are equally important and have the greatest impact on alignment performance.

* **Inspect the Peaks:** Use `evaluation/ip_metrics_and_viz.py` to visualize the detected peaks and confirm that they provide sufficient spatial coverage across the dataset.

### 2. Matching

The key matching metrics are correspondence quality, match quantity, spatial coverage, and the residual error between matched points after alignment.

* **Rigid Matching:** Because the rigid model has fewer degrees of freedom, you can generally allow more matches to produce a robust estimate of translation and rotation.

* **Affine Matching:** Match quality becomes more important because incorrect correspondences can introduce unwanted scaling, shear, or deformation. Stricter filtering is typically required.

* **Split-Affine Matching:** Spatial coverage is critical. Ensure that each split contains enough real correspondences and synthetic points to remain connected and well constrained during optimization.

* **Inspect the Matches:** Use `evaluation/match_viz.py` to visualize the matches. Confirm that they span the regions being aligned and provide sufficient coverage throughout the dataset.

### 3. Solver

The solver iteratively optimizes the alignment through the rigid, affine, and split-affine stages. Each stage begins from the result of the previous stage and introduces additional model flexibility.

During optimization, the solver can remove poor correspondences based on their residual error. Use the absolute and relative error thresholds to control this outlier rejection. If the maximum error is significantly higher than the average error, the thresholds may need to be stricter.
  
---

<br>

## Build Package

### Using the Built `.whl` File

1. **Build the source distribution and `.whl` file from the repository root:**
  ```sh
  cd /path/to/Rhapso
  python -m pip install --upgrade build
  python -m build
  ```
  The distributions will appear in the `dist` directory. Do not rename the wheel because its filename contains package compatibility metadata (for example, `rhapso-<version>-py3-none-any.whl`).

---

<br>
<br>
<br>
