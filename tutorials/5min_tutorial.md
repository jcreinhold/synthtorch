# Quick Tutorial

## Install package

The easiest way to install the package is with the following command:

    pip install git+git://github.com/jcreinhold/synthnn.git
    
You can also download the package through git, i.e.,

    git clone https://github.com/jcreinhold/synthnn.git

then you can install the package via the setup.py script, i.e.,
inside the `synthnn` directory, run the following command:

    python setup.py install

or 

    python setup.py develop
    
If you prefer Docker or Singularity, use one of the following commands:

    docker pull jcreinhold/synthnn

or 

    singularity pull docker://jcreinhold/synthnn


## Deep Neural Network-based Synthesis

To use a deep neural network via pytorch for synthesis, we provide an example call to the training routine:

```bash
nn-train -s t1/ \
         -t flair/ \
         --output model_dir/unet.pth \
         --nn-arch unet \
         --n-layers 3 \
         --n-epochs 100 \
         --patch-size 64 \
         --batch-size 1 \
         --n-jobs 0 \
         --plot-loss loss_test.png \
         -vv \
         --out-config-file config.json 
``` 

Note the `--out-config-file` argument which creates a json file which contains all the experiment configurations.
We can then use the following command to run addition training as follows:

```bash
nn-train config.json
```

You can edit the config.json file directly to edit experiment parameters, and this is the preferred interface for using
the neural network synthesis routines.

Interact with `nn-predict` by generating a configuration file as shown above. Then edit
the prediction directory parameters (i.e., the `predict_out` and `predict_dir` fields) in the file, 
and then run the command:

```bash
nn-predict config.json
```

Note that when you generate a configuration file, there will be brackets around the training and validation
directories used. Also use brackets around your directory/directories that you input into the `predict_dir` field.
