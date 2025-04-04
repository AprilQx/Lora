Introduction
===========

Background
---------

Numerical forecasting is a critical task across many domains, from financial markets to climate science. 
Traditional approaches rely on statistical methods and domain-specific models. Recent advances in large 
language models (LLMs) have shown promise in handling numerical tasks despite being primarily designed 
for text processing.

Problem Statement
--------------

This project addresses the question: **Can LLMs effectively perform numerical time series forecasting?** 
Specifically, we investigate:

1. How well can language models forecast predator-prey population dynamics?
2. Can parameter-efficient fine-tuning methods adapt these models for numerical tasks?
3. What is the computational efficiency of such approaches?

Approach
-------

We focus on the Lotka-Volterra equations, a classic model of predator-prey dynamics that produces 
oscillatory patterns. Our approach includes:

* Representing numerical time series as text sequences
* Implementing Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
* Comprehensive evaluation against statistical baselines
* Analysis of computational efficiency through FLOP tracking

Key Contributions
--------------

This project demonstrates:

* The feasibility of adapting LLMs for numerical forecasting with minimal parameter updates
* Insights into optimal context lengths and text formatting for numerical data
* A framework for tracking computational usage during model adaptation and inference

.. [1] Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). Large Language Models Are Zero-Shot Time Series Forecasters. In Neural Information Processing Systems (NeurIPS 2023).