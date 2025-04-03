Methodology
==========

Model Training Approach
----------------------

Our approach involves fine-tuning pre-trained language models (specifically Qwen2.5) using Low-Rank Adaptation (LoRA) for numerical forecasting tasks.

.. figure:: _static/methodology_diagram.png
   :alt: Training methodology diagram
   :width: 700px
   
   Overview of the model training and evaluation pipeline.

Parameter-Efficient Fine-Tuning
------------------------------

We use LoRA to modify only a small subset of model parameters:

* Rank: r=8
* Scaling factor: α=16
* Target modules: Query and Value projection matrices
* Total trainable parameters: <1% of full model

Evaluation Framework
------------------

Models are evaluated on the following metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Maximum Error
* Computational efficiency (FLOPs)