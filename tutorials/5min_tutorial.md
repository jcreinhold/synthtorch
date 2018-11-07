# Quick Tutorial

## Install package

The easiest way to install the package is with the following command:

    pip install git+git://github.com/jcreinhold/synthnn.git
    
You can also download the package through git, i.e.,

    git clone https://github.com/jcreinhold/synthnn.git

then you can try to install the package via the setup.py script, i.e.,
inside the `synthnn` directory, run the following command:

    python setup.py install

or 

    python setup.py develop
    
If you don't want to bother with any of this, you can create a Docker image or Singularity image via:

    docker pull jcreinhold/synthnn

or 

    singularity pull docker://jcreinhold/synthnn


## Deep Neural Network-based Synthesis

To use a deep neural network via pytorch for synthesis, we provide an example call to the training routine:

```bash
nn-train -s t1/ \
         -t flair/ \
         --output model_dir/unet.pkl" \
         --nn-arch unet \
         --n-layers 3 \
         --n-epochs 100 \
         --patch-size 64 \
         --batch-size 1 \
         --n-jobs 0 \
         --validation-count 1 \
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

You can either call `nn-predict` as in the first example with the relevant parameters filled in (see the `-h` option to 
view all the commands). The preferred way to interact with `nn-predict` is to generate a configuration file, edit
the prediction directory parameters in the file (which should only consist of setting the directory on which to do synthesis
and set the directory to output the results), and then run the command:

```bash
nn-predict config.json
```
