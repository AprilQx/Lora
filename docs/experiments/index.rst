Experiments
==========

This section documents the experiments conducted in our project.

.. toctree::
   :maxdepth: 2

   baseline_evaluation
   lora_finetuning
   hyperparameter_search
   context_length

Overview
--------

Our experimental framework includes:

* **Baseline Evaluation**: Testing the unmodified Qwen2.5 model on numerical forecasting tasks
* **LoRA Fine-tuning**: Parameter-efficient adaptation of the model for time series prediction
* **Hyperparameter Search**: Systematic exploration of learning rates, LoRA ranks, and precision values
* **Context Length Exploration**: Analysis of how input context length affects prediction accuracy

Experimental Setup
----------------

All experiments were conducted with:

* Hardware: NVIDIA A100 GPU
* FLOP Budget: 1e17 FLOPs (tracked for each experiment)
* Dataset: Lotka-Volterra predator-prey time series
* Base Model: Qwen2.5-0.5B-Instruct

