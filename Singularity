Bootstrap: docker
From: continuumio/miniconda3

%environment

    # use bash as default shell
    SHELL=/bin/bash
    export SHELL

    # add CUDA paths
    CPATH="/usr/local/cuda/include:$CPATH"
    PATH="/usr/local/cuda/bin:$PATH"
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    CUDA_HOME="/usr/local/cuda"
    export CPATH PATH LD_LIBRARY_PATH CUDA_HOME

    # make conda environment accessible
    PATH=/opt/conda/bin:$PATH
    export PATH

%post

    # load environment variables
    . /environment

    # make environment file executable
    chmod +x /environment

    # default mount paths, files
    mkdir /scratch /data /work-zfs
    touch /usr/bin/nvidia-smi

    # install package
    git clone https://github.com/jcreinhold/synthnn
    cd synthnn
    /opt/conda/bin/conda install --override-channels -c pytorch -c defaults python=3.7 numpy matplotlib pytorch torchvision cuda92 --yes
    /opt/conda/bin/conda install --override-channels -c conda-forge nibabel scikit-image --yes
    /opt/conda/bin/pip install git+git://github.com/jcreinhold/niftidataset.git
    /opt/conda/bin/pip install git+git://github.com/jcreinhold/synthqc.git
    /opt/conda/bin/pip install git+git://github.com/NVIDIA/apex.git
    /opt/conda/bin/pip install --upgrade setuptools
    /opt/conda/bin/pip install -e .

%runscript
    exec python $@

%apprun train
    exec nn-train $@

%apprun predict
    exec nn-predict $@
