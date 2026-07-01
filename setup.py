from setuptools import setup, find_packages
from pathlib import Path   

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='Rhapso',
    version='0.2.8',
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
        'pandas',
        'PyYAML==6.0.2',
        'numcodecs>=0.14',
        'scipy==1.13.1',
        'scikit-image',
        'matplotlib==3.10.0',
        'memory-profiler==0.61.0',
        'scikit-learn',
        'click==8.2.1',
        'dask[array]==2024.12.1',
        'dask-image==2024.5.3',
        'zarr[remote]>=3.0.8,<3.2',
        'bioio==1.3.0',
        'bioio-tifffile==1.0.0',
        'tifffile==2025.1.10',
        'ome-zarr',
        's3fs==2024.12.0',
        'boto3==1.35.92',
        'ray[default]==2.9.1',
    ],
    python_requires='>=3.11',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
)
