Hyperparameter Search
===================

Overview
--------

We conducted a comprehensive grid search to find optimal hyperparameters for our LoRA fine-tuning approach. This search explored variations in learning rates, LoRA ranks, and numeric precision.

Search Space
----------

The following hyperparameters were explored:

* **Learning rates**: 1e-5, 5e-5, 1e-4
* **LoRA ranks**: 2, 4, 8
* **Precision values**: 2, 3 (decimal places)

This resulted in 18 different model configurations, with each model trained for up to 2,000 steps.

Methodology
----------

For each configuration:

1. A fresh model instance was initialized
2. LoRA was applied with the specified rank
3. Training proceeded for up to 2,000 steps
4. The model was evaluated on the validation set every 500 steps
5. The best validation MAE was recorded
6. FLOP usage was tracked for computational efficiency analysis

Key Findings
----------

* **Best Configuration**:
  * Learning rate: 1e-4
  * LoRA rank: 8
  * Precision: 2 decimal places
  * Validation MAE: 0.209
  
* **Parameter Impact**:
  * Higher learning rates (1e-4) generally performed better than lower ones
  * Performance improved with larger LoRA ranks
  * Precision of 2 decimal places was slightly better than 3 decimal places

.. figure:: ../../results/hyperparameter_search/combined_heatmap.png
   :alt: Hyperparameter Search Results
   :width: 700px
   
   Heatmap showing validation MAE for different hyperparameter combinations.

Comparison of Precision Values
---------------------------

.. figure:: ../../results/hyperparameter_search/precision_effect.png
   :alt: Effect of Precision on MAE
   :width: 600px
   
   Boxplot showing the effect of precision on validation MAE.

FLOP Efficiency
-------------

Configurations with higher LoRA ranks used more FLOPs, but the increase in computational cost was offset by substantial improvements in prediction accuracy.