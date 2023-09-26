# Enables `pip install` for our package #

from setuptools import setup
from setuptools import find_packages

# list dependencies from file
with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(
    name="neet",
    version="1.0.3",
    description="package for NEET project",
    packages=find_packages(),  # NEW: find packages automatically
    install_requires=requirements,
    package_data={
        "assets.models": ["*.pkl"],
        "assets.templates": ["*.csv"],
    },
)  # NEW
