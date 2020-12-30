# Deep learning halos

This repository contains all the relevant code used in Lucie-Smith et al. 2020, "Deep learning insights into cosmological structure formation", https://ui.adsabs.harvard.edu/abs/2020arXiv201110577L/abstract.

A CNN takes as input sub-regions of the initial conditions of N-body simulations and predicts the mass of the resulting dark matter halos at z=0.

- `dlhalos_code`: contains most important Python modules involving the main steps of the pipeline: data processing, training and evaluation.
- `scripts`: contains the scripts used to produce the results in the paper.
- `plots` & `paper_plots`: functions to make general plots and those used in paper from the outputs.
- `nose`: unit tests.
- `utilss`: contains useful funcitons used throughout the scripts.
- `scratch`: contains a large number of test runs used during production.
