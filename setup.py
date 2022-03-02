#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    install_requires=["pytorch-lightning"],
    packages=find_packages(),
    python_requires="~=3.8",
)
