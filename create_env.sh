#!/bin/bash
# use the following command to run this script: . ./create_env.sh
# Created on: Nov 8, 2018
# Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

FASTAI=false
if [[ "$1" == "--fastai" ]]; then
  FASTAI=true
fi

if [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "darwin"* ]]; then
    :
else
    echo "Operating system must be either linux or OS X"
    return 1
fi

command -v conda >/dev/null 2>&1 || { echo >&2 "I require anaconda but it's not installed.  Aborting."; return 1; }

# first make sure conda is up-to-date
conda update -n base conda --yes

# define all the packages needed
packages=(
    numpy
    matplotlib
    scikit-learn
    scipy
)

fastai_packages=(
    fastai==1.0.41
)

# assume that linux is GPU enabled (except for in CI) but OS X is not
ONTRAVIS=${TRAVIS:-false}

if [[ "$OSTYPE" == "linux-gnu" && "$ONTRAVIS" == false ]]; then
    pytorch_packages=(
        pytorch
        torchvision
        cudatoolkit=9.0
    )
else
    pytorch_packages=(
        pytorch-cpu
        torchvision-cpu
    )
fi

conda_forge_packages=(
    nibabel
    scikit-image
)

# create the environment and switch to that environment

if $FASTAI; then
    echo "conda create --name synthnn --override-channels -c pytorch -c fastai -c defaults ${packages[@]} ${fastai_packages[@]} --yes"
    conda create --name synthnn --override-channels -c pytorch -c fastai -c defaults ${packages[@]} ${fastai_packages[@]} --yes
else
    echo "conda create --name synthnn --override-channels -c pytorch -c defaults ${packages[@]} ${pytorch_packages[@]} --yes"
    conda create --name synthnn --override-channels -c pytorch -c defaults python=3.7 ${packages[@]} ${pytorch_packages[@]} --yes
fi

source activate synthnn
# add a few other packages
conda install -c conda-forge ${conda_forge_packages[@]} --yes 
pip install git+git://github.com/jcreinhold/niftidataset.git
pip install git+git://github.com/jcreinhold/synthqc.git
pip install git+git://github.com/NVIDIA/apex.git

# install this package
python setup.py develop

echo "synthnn conda env script finished (verify yourself if everything installed correctly)"
