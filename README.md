# Deep learning halos

This repository contains the code used in Lucie-Smith, Peiris, Pontzen, Nord, Thiyagalingam, ["Deep learning insights into cosmological structure formation"](https://ui.adsabs.harvard.edu/abs/2020arXiv201110577L/abstract), 2020, to learn about dark matter halo formation with convolutional neural networks.

For those wanting to try out our code, the best place to start is the ipython notebook demo. Please see below for instructions.

If you use this dataset in your work, please cite it as follows:

## Bibtex
```
@ARTICLE{2020arXiv201110577L,
       author = {{Lucie-Smith}, Luisa and {Peiris}, Hiranya V. and {Pontzen}, Andrew and {Nord}, Brian and {Thiyagalingam}, Jeyan},
        title = "{Deep learning insights into cosmological structure formation}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Artificial Intelligence, Computer Science - Machine Learning},
         year = 2020,
        month = nov,
          eid = {arXiv:2011.10577},
        pages = {arXiv:2011.10577},
archivePrefix = {arXiv},
       eprint = {2011.10577},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv201110577L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Description

The CNN predicts the final mass of dark matter halos given the initial conditions of a cosmological simulation.
The CNN takes as input a cubic sub-region of the initial conditions, centred on a given simulation particle, and predicts the mass of the dark matter halo to which that particle will belong at z=0.

- `dlhalos_code`: contains most important Python modules involving the main steps of the pipeline: data processing, training and evaluation.
- `scripts`: contains the scripts used to produce the results in the paper.
- `plots` & `paper_plots`: functions to make general plots and those used in paper from the outputs.
- `nose`: unit tests.
- `utilss`: contains useful funcitons used throughout the scripts.
- `scratch`: contains a large number of test runs used during production.

## Software dependencies

The code requires pre-installation of the following software: standard Python packages, such as numpy, scipy, and matplotlib;[Tensorflow 1.14](https://www.tensorflow.org); [pynbody](https://pynbody.github.com/pynbody/); [numba](https://numba.pydata.org); [scikit-learn](https://scikit-learn.org).

## Demo Jupyter Notebook

The repository includes a demo that demonstrates t. You can open the `demo_script.ipynb` file in `demo` directory using Jupyter notebook. This is what you need:

1. First, you need a working version of python3, which contains Tensorflow 1.14 and all other software dependencies.

2. You will also need to download the data from [Google Cloud Storage](https://console.cloud.google.com/storage/browser/deep-halos-data?cloudshell=false&hl=en-AU&project=deephalos&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false). Click on this link and download the `demo-data` folder. Make sure that the `demo-data` folder is located at the directory where the notebook is running (e.g. inside the `demo` directory). Note that the data we have provided here is only a small subset of that used in the paper, but is enough to familiarize with the code.

Once you are done with these steps, you're good to go. You can run through the ipython notebook and predict final halo masses from the initial conditions! The notebook should run within a few minutes on your laptop -- no need for GPUs. You can also make changes in the parameter file `params_demo.py` to play around with different choices for the CNN architecture, the training set size, the size of the cubic sub-region inputs, and so on.

## Results

To reproduce the results in ["Deep learning insights into cosmological structure formation"](https://ui.adsabs.harvard.edu/abs/2020arXiv201110577L/abstract), Lucie-Smith, Peiris, Pontzen, Nord, Thiyagalingam, 2020, one should use the code in the `scripts` directory. Each CNN model described in the paper has its own sub-directory, which includes the parameter file and two scripts, one for training and one for evaluation. 

To run these, you will also need to download the full dataset from [Google Cloud Storage](https://console.cloud.google.com/storage/browser/deep-halos-data/full-data?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&cloudshell=false&hl=en-AU&project=deephalos&prefix=&forceOnObjectsSortingFiltering=false).
