# Use official python runtime as a parent image
FROM continuumio/miniconda3
MAINTAINER Jacob Reinhold, jacob.reinhold@jhu.edu

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# 1) Create a conda environment will all required dependencies
# 2) Install this package into the container
# 3) Setup matplotlib to not pull in a GUI
# 4) Set Docker to automatically start the created env
RUN /bin/bash -c "source ./create_env.sh" && \
    python setup.py install && \
    echo "backend: agg" > matplotlibrc && \
    sed -i '/conda activate base/d' ~/.bashrc && \
    echo "conda activate synthtorch" >> ~/.bashrc
