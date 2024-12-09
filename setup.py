# setup.py
from setuptools import setup, find_packages

setup(
    name="Rhapso",
    version="0.0.1",
    author="Martin Barker, Sean Fite",
    author_email="martin.barker@alleninstitute.org, sean.fite@alleninstitute.org",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'Rhapso=Rhapso:main',
        ],
    },
)