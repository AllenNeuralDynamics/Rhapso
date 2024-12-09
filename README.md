# Rhapso

Rhapso is a Python-based tool designed to robustly and efficiently align and fuse large-scale microscopy datasets. The tool is built with flexibility, scalability, and modularity in mind, making it suitable for cloud-native deployments, as well as on-prem or cluster-based executions. It aims to improve existing imaging pipelines by incorporating performance optimizations, enhanced robustness, and automation through machine learning.

Rhapso is being developed as part of the Allen Institute for Neurotechnology (AIND) and will be published as an open-source software component in the OCTO SDK for image processing. Initially, it will benefit AIND's ExaSPIM pipeline and the broader scientific community in their microscopy research.

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)

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
print(Rhapso.say_hello("Test"))
```

### Uninstall Rhapso Package
```sh
pip uninstall rhapso
```

### To Do:
- Setup and add Tests 
- Improve `setup.py` to include more metadata details about this package.