# Use official python runtime as a parent image
FROM python:3.6-stretch
MAINTAINER Jacob Reinhold, jacob.reinhold@jhu.edu

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# 1) Install any needed packages specified in requirements.txt
# 2) Install this package into the container
# 3) Setup matplotlib to not pull in a GUI
# 4) Install apex for mixed precision
RUN pip install --upgrade pip && \
    pip install --trusted-host pypi.python.org -r requirements.txt && \
    python setup.py install && \
    echo "backend: agg" > matplotlibrc && \
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    python setup.py install && \
    cd ..
