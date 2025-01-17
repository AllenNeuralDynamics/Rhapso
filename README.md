# Rhapso

Rhapso is a Python-based tool designed to robustly and efficiently align and fuse large-scale microscopy datasets. The tool is built with flexibility, scalability, and modularity in mind, making it suitable for cloud-native deployments, as well as on-prem or cluster-based executions. It aims to improve existing imaging pipelines by incorporating performance optimizations, enhanced robustness, and automation through machine learning.

Rhapso is being developed as part of the Allen Institute for Neurotechnology (AIND) and will be published as an open-source software component in the OCTO SDK for image processing. Initially, it will benefit AIND's ExaSPIM pipeline and the broader scientific community in their microscopy research.

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)

## Table of Contents
- [Repository Structure](#repository-structure)
- [Command Line Usage](#command-line-usage)
- [Setup Instructions](#setup-instructions)
- [Build Package](#build-package)
  - [Using the Built `.whl` File](#using-the-built-whl-file)
- [Run Tests](#run-tests)
- [Environments](#environments)
- [Use Cases](#use-cases)
- [To Do](#to-do)

---

## Repository Structure

```
rhapso/
│
├── Rhapso/                      # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # Command-line interface entry point
│
│   ├── detection/               # Detection algorithms
│   │   ├── __init__.py
│   │   └── interest_points.py   # Interest point detection
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
│   │   └── solver.py            # Solve transformations
│
├── tests/                       # Unit tests for each module
│   ├── test_detection.py
│   ├── test_fusion.py
│   ├── test_matching.py
│   ├── test_solving.py
│   └── __init__.py
│
├── setup.py                     # Package installation
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies
```

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
Rhapso match --method ORB --distance 0.7 --verbose
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

### To Do:
- Setup and add Tests 
- Improve `setup.py` to include more metadata details about this package.
