Data
====

This section documents the datasets and preprocessing methods used in our project.

.. toctree::
   :maxdepth: 2

   dataset
   preprocessing
   exploratory_analysis

Overview
--------

Our project uses a Lotka-Volterra dataset, which contains predator-prey population dynamics over time. The data consists of:

* 1,000 trajectory samples
* 100 time points per trajectory
* 2 variables per time point (prey and predator populations)

Key features of our data pipeline:

* Data formatting for language model input
* Train/validation/test splitting (70/15/15)
* Text-based representation of numerical values/Users/apple/Documents/GitLab_Projects/M2_coursework/results/data_analysis