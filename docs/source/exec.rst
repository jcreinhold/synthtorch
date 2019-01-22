Executables
===================================

Neural Network Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: synthnn.exec.nn_train
   :func: arg_parser
   :prog: nn-train

Neural Network Predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The prediction function only supports the use of a configuration file
(which can be created from the use of the `nn-train`, see the `--out-config-file` option).
This is due to pytorch requiring the parameters to recreate the neural network class, which
can then be updated with the trained weights.

Note that you will have to change the `predict_out` and `predict_dir` fields in the .json file
with where the output files should be stored and where the source images should come from, respectively.

There may be other fields that need to be altered based on your specific configuration.
