#!/usr/bin/env python3

import setuptools
import versioneer
from pathlib import Path


install_requires = [
    "pandas>=1.5.0",
    "pytest",
    "requests",
    "tqdm",
    "numpy",
    "click",
    "psutil",
    "jupyterlab",
    "filelock",
    "ffmpeg-python"
]


with open(Path(__file__).parent.joinpath("README.md")) as f:
    readme = f.read()


classifiers = \
    [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Science/Research"
    ]


setuptools.setup(
    name="lbm_mc",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="High level pandas-based API for batch analysis of Calcium Imaging data using CaImAn",
    long_description=readme,
    long_description_content_type='text/markdown',
    classifiers=classifiers,
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    include_package_data=True,
    url="https://github.com/nel-lab/mesmerize-core",
    license="Apache-Software-License",
    author="Kushal Kolar, Caitlin Lewis, Arjun Putcha",
    author_email="",
)
