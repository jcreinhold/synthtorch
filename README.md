synthtorch
=======================

[![Build Status](https://api.travis-ci.com/jcreinhold/synthtorch.svg?branch=master&status=passed)](https://travis-ci.com/github/jcreinhold/synthtorch)
[![Coverage Status](https://coveralls.io/repos/github/jcreinhold/synthtorch/badge.svg?branch=master)](https://coveralls.io/github/jcreinhold/synthtorch?branch=master)
[![Documentation Status](https://readthedocs.org/projects/synthtorch/badge/?version=latest)](http://synthtorch.readthedocs.io/en/latest/)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/jcreinhold/synthtorch.svg)](https://hub.docker.com/r/jcreinhold/synthtorch/)
[![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2669612.svg)](https://doi.org/10.5281/zenodo.2669612)

**This package may not work with current versions of PyTorch and will not be supported.**

This package contains deep neural network-based (pytorch) modules to synthesize magnetic resonance (MR) and computed 
tomography (CT) brain images. Synthesis is the procedure of learning the transformation that takes a specific contrast image to another estimate contrast.

For example, given a set of T1-weighted (T1-w) and T2-weighted (T2-w) images, we can learn the function that maps the intensities of the
T1-w image to match that of the T2-w image via a UNet or other deep neural network architecture. In this package, we supply 
the framework and several models for this type of synthesis. See the `Relevant Papers` section (at the bottom of 
the README) for a non-exhaustive list of some papers relevant to the work in this package.

We also support a *non*-DNN-based synthesis package called [synthit](https://gitlab.com/jcreinhold/synthit).
There is also a seperate package to gather quality metrics of the synthesis result called [synthqc](https://gitlab.com/jcreinhold/synthqc).

** Note that this is an **alpha** release. If you have feedback or problems, please submit an issue (it is very appreciated) **

This package was developed by [Jacob Reinhold](https://jcreinhold.github.io) and the other students and researchers of the 
[Image Analysis and Communication Lab (IACL)](http://iacl.ece.jhu.edu/index.php/Main_Page).

[Link to main Gitlab Repository](https://gitlab.com/jcreinhold/synthtorch)

Requirements
------------

- matplotlib
- nibabel >= 2.3.1
- [niftidataset](https://github.com/jcreinhold/niftidataset) >= 0.1.4
- numpy >= 1.15.4
- pillow >= 5.3.0
- torch >= 1.2.0
- torchvision >= 0.2.1

Installation
------------

    pip install git+git://github.com/jcreinhold/synthtorch.git

Tutorial
--------

[5 minute Overview](https://github.com/jcreinhold/synthtorch/blob/master/tutorials/5min_tutorial.md)

[Jupyter Notebook example](https://nbviewer.jupyter.org/github/jcreinhold/synthtorch/blob/master/tutorials/tutorial.ipynb)

In addition to the above small tutorial and example notebook, there is consolidated documentation [here](https://synthtorch.readthedocs.io/en/latest/).

Singularity
-----------

You can build a singularity image from the docker image hosted on [dockerhub](https://hub.docker.com/r/jcreinhold/synthtorch/) or through [singularity-hub](https://www.singularity-hub.org/collections/2909) via the following command:

    singularity pull shub://jcreinhold/synthtorch:latest

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v tests

Citation
--------

If you use the `synthtorch` package in an academic paper, please use the following citation:

    @misc{reinhold2019,
        author       = {Jacob Reinhold},
        title        = {{synthtorch}},
        year         = 2019,
        doi          = {10.5281/zenodo.2669612},
        version      = {0.3.2},
        publisher    = {Zenodo},
        url          = {https://doi.org/10.5281/zenodo.2669612}
    }
    
Relevant Papers
---------------

[1] C. Zhao, A. Carass, J. Lee, Y. He, and J. L. Prince, “Whole Brain Segmentation and Labeling from CT Using Synthetic MR Images,” in MICCAI MLMI, vol. 10541, pp. 291–298, 2017.
