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

    singularity pull shub://jcreinhold/synthnn:latest


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
We can then use the following command to run additional training as follows:

```bash
nn-train config.json
```

You can edit the config.json file directly to edit experiment parameters, and this is the preferred interface for using
the neural network synthesis routines. [Here](https://gist.github.com/jcreinhold/793d26387f6a8b9f6966b59c6705f249) is
an example configuration file.

Interact with `nn-predict` by generating a configuration file as shown above. Then edit
the prediction directory parameters (i.e., the `predict_out` and `predict_dir` fields) in the file, 
and then run the command:

```bash
nn-predict config.json
```

Note that when you generate a configuration file, there will be brackets around the training and validation
directories used. Also use brackets around your directory/directories that you input into the `predict_dir` field.

## Example training/validation directory setup

Below is an example of a training/validation directory to be used with synthnn. 

```none
├── flair
│   ├── train
│   │   ├── subject1_flair.nii.gz
│   │   ├── subject2_flair.nii.gz
│   │   ├── subject3_flair.nii.gz
│   │   ├── ...
│   │
│   └── valid
│       ├── subjecta_flair.nii.gz
│       ├── subjectb_flair.nii.gz
│       ├── subjectc_flair.nii.gz
│       ├── ...
│   
├── t1
│   ├── train
│   │   ├── subject1_t1.nii.gz
│   │   ├── subject2_t1.nii.gz
│   │   ├── subject3_t1.nii.gz
│   │   ├── ...
│   │
│   └── valid
│       ├── subjecta_t1.nii.gz
│       ├── subjectb_t1.nii.gz
│       ├── subjectc_t1.nii.gz
│       ├── ...
│   
└── t2
    ├── train
    │   ├── subject1_t2.nii.gz
    │   ├── subject2_t2.nii.gz
    │   ├── subject3_t2.nii.gz
    │   ├── ...
    │
    └── valid
        ├── subjecta_t2.nii.gz
        ├── subjectb_t2.nii.gz
        ├── subjectc_t2.nii.gz
        ├── ...
```

Note that the directory structure 
isn’t that important though, since you specify in the config file or in the command-line the training and validation 
directories (or just the training if you want to use the `–valid-split` option). The important part shown in the 
below directory tree is that _each subject has a common naming convention across all input modalities_. 
You see that in `t1/train`, `t2/train`, and `flair/train`, the subjects all have the same naming convention, such that 
when the images are alphabetically sorted in each directory respectively, they align. That is, the first element 
of the sorted `t1/train` aligns with the first element of `t2/train` aligns with the first element of `flair/train` 
and so on. That is how the data loader grabs files and processes them, so this setup is absolutely required.
 
An important note is that, since the synthesis method used is a supervised method. So it is required that all of 
the subject scans be co-registered (e.g., FLAIR and T2 are registered to T1). Additionally, all of the images must 
be of the same size (e.g., if the T1-w images are of dimension h _x_ w _x_ d, then the T2-w and FLAIR are also of 
dimensions h _x_ w _x_ d).

## Example testing directory setup

An example directory for prediction in the case of multi-modal synthesis \{T1,T2\}-to-FLAIR 
(with example results) would be:

```none
├── synthesized
│   └── results
│       ├── subjectx_synth_flair.nii.gz
│       ├── subjecty_synth_flair.nii.gz
│       ├── subjectz_synth_flair.nii.gz
│       └── ...
│   
├── t1
│   └── test
│       ├── subjectx_t1.nii.gz
│       ├── subjecty_t1.nii.gz
│       ├── subjectz_t1.nii.gz
│       └── ...
│   
└── t2
    └── test
        ├── subjectx_t2.nii.gz
        ├── subjecty_t2.nii.gz
        ├── subjectz_t2.nii.gz
        └── ...
```

The reason for making this explicit will be motivated in the 2D synthesis section.

## _n_-to-_m_ Multi-modal Synthesis

As alluded to in the previous section, it is worth specifically noting that the synthnn package (specifically 
`nn-train` and `nn-predict`) supports multi-modal image synthesis. This is done by simply adding additional 
directories to the source or target arguments, e.g. for \{T1,T2\}-to-FLAIR synthesis, we have:

```bash
nn-train -s t1/ t2/ \
         -t flair/ \
         ...
``` 

and for T1-to-\{T2,FLAIR\} synthesis, we have

```bash
nn-train -s t1/ \
         -t flair/ t2/ \
         ...
``` 

## 2D Synthesis Support

`nn-train` also supports using TIFF images as training data instead of NIfTI files. This is actually the preferred
method of using the package when doing 2D synthesis, since this is the easiest way to iterate over the entire 
training dataset. To convert your NIfTI files to a set of TIFF files, you can use [this script](https://gist.github.com/jcreinhold/01daf54a6002de7bd8d58bad78b4022b)
which takes a directory of NIfTI files and outputs a numbered order of TIFF files corresponding to image slices.
Note that the images may _appear_ empty if you open with a standard image viewer, but they actually do contain
slices (try to plot it with python or open it in MIPAV if that is a software you are familiar with).

If we took the previously shown example as shown in the **Example training/validation directory setup** section
and applied the previously discussed script to the images, then an example directory structure with TIFF images
would be the following:

```none
├── flair
│   ├── train
│   │   ├── subject1_flair_100.tif
│   │   ├── subject1_flair_101.tif
│   │   ├── subject1_flair_102.tif
│   │   ├── ...
│   │
│   └── valid
│       ├── subjecta_flair_100.tif
│       ├── subjecta_flair_101.tif
│       ├── subjecta_flair_102.tif
│       ├── ...
│   
├── t1
│   ├── train
│   │   ├── subject1_t1_100.tif
│   │   ├── subject1_t1_101.tif
│   │   ├── subject1_t1_102.tif
│   │   ├── ...
│   │
│   └── valid
│       ├── subjecta_t1_100.tif
│       ├── subjecta_t1_101.tif
│       ├── subjecta_t1_102.tif
│       ├── ...
│   
└── t2
    ├── train
    │   ├── subject1_t2_100.tif
    │   ├── subject1_t2_101.tif
    │   ├── subject1_t2_102.tif
    │   ├── ...
    │
    └── valid
        ├── subjecta_t2_100.tif
        ├── subjecta_t2_101.tif
        ├── subjecta_t2_102.tif
        ├── ...
```

Note that the same assumptions about data applies in this setup as before (e.g., TIFF images must be in correspondence to 
the same subject/slice across directories).

Finally, to do prediction with 2D synthesis, note that you would use a testing directory structure as shown in 
the **Example testing directory setup** section. That is, `nn-predict` expects NIfTI files regardless of whether TIFF
images were used for training! The reason for this is that we want to reuse the test NIfTI file's header for
the synthesized product. This is precisely the reason why I suggest the TIFF image format, because it supports
`float32` data types and thus allows you to train on single slices images which contain the exact same type of
information that is present in a NIfTI file.

## Final note

If this tutorial was unclear in any way or if you are having trouble with the package, please open an issue or email
me with your question. I'm very happy to receive feedback.
