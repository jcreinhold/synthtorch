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

version = '0.1.4'

args = dict(
    name='synthnn',
    version=version,
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
                            'nn-predict=synthnn.exec.nn_predict:main']
    },
    dependency_links=[f'git+git://github.com/jcreinhold/niftidataset.git@master#egg=niftidataset-{version}']
)

setup(install_requires=['matplotlib',
                        'nibabel',
                        'niftidataset',
                        'numpy',
                        'torch',
                        'torchvision'], **args)
