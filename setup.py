#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Module installs the synthnn package
Can be run via command: python setup.py install (or develop)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='synthnn',
    version='0.1.0',
    description="pytorch-based image synthesis",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://gitlab.com/jcreinhold/synthnn',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'tutorials')),
    keywords="mr image-synthesis",
    entry_points={
        'console_scripts': ['nn-train=synthnn.exec.nn_train:main',
                            'nn-predict=synthnn.exec.nn_predict:main', ]
    },
    dependency_links=['git+git://github.com/jcreinhold/niftidataset.git']
)

setup(install_requires=['nibabel',
                        'numpy',
                        'torch',
                        'torchvision'], **args)
