from setuptools import setup, find_packages
from pathlib import Path   

# read README.md for the long project description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='Rhapso',
    version='0.2.5',
    author='ND',
    author_email='sean.fite@alleninstitute.org',
    description='A python package for aligning and stitching light sheet fluorescence microscopy images',
    long_description=long_description,                 
    long_description_content_type='text/markdown', 
    project_urls={
        'Source': 'https://github.com/AllenNeuralDynamics/Rhapso',
        'Roadmap': 'https://github.com/AllenNeuralDynamics/Rhapso/issues',
    },
    packages=find_packages(),
    install_requires=[
        'pandas==3.0.2',
        'dask[array]==2024.12.1',
        'zarr==2.18.3',
        'scipy==1.13.1',
        'scikit-image==0.22.0',
        'bioio==1.3.0',
        'bioio-tifffile==1.0.0',
        'tifffile==2025.1.10',
        'dask-image==2024.5.3',
        'boto3==1.35.92',
        'numcodecs==0.13.1',
        'matplotlib==3.10.0',
        'memory-profiler==0.61.0',
        's3fs==2024.12.0',
        'scikit-learn==1.8.0',
        'click==8.2.1',

        'ray==2.54.1',
        'tensorstore==0.1.82',
        'xmltodict==1.0.4',
        'nptyping==2.5.0',
        "setuptools==71.0.4"
    ],
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
)













