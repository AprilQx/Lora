# ML Forecasting Project

A project for fine-tuning and evaluating language models on numerical forecasting tasks using Low-Rank Adaptation (LoRA).

## Overview

This project explores the capabilities of foundation models like Qwen2.5 for numerical time-series forecasting. 
We implement parameter-efficient fine-tuning approaches and evaluate model performance on predator-prey dynamics.

### Key Contributions

* Demonstrating the feasibility of adapting LLMs for numerical forecasting with minimal parameter updates
* Insights into optimal context lengths and text formatting for numerical data
* A framework for tracking computational usage during model adaptation and inference

## Installation

### Requirements

* Python 3.9+
* PyTorch 2.6+
* CUDA-compatible GPU (recommended)
* 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://gitlab.com/your-username/M2_coursework.git
cd M2_coursework

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download and process the data
python src/data/download_data.py
python src/data/process_data.py
```
## Methodology
Our approach involves fine-tuning pre-trained language models (specifically Qwen2.5) using Low-Rank Adaptation (LoRA) for numerical forecasting tasks. The methodology includes:
1. **Base Model**: Start with Qwen2.5-0.5B-Instruct model
2. **Hyperparameter Search**: Test variations in learning rates, LoRA ranks, and precision
3. **Context Length Study** Evaluate different input lengths (128, 512, 768 tokens)
4. **Final Model**: Train with optimal hyperparameters (LR=1e-4, Rank=8, Context=512)
5. **Evaluation**: Assess performance using MAE, RMSE, and FLOP tracking

We use LoRA to modify only a small subset of model parameters:

* Rank: r=8
* Scaling factor: α=16
* Target modules: Query and Value projection matrices
* Total trainable parameters: <1% of full model


## Dataset
We use the Lotka-Volterra dataset that models predator-prey population dynamics:
* 1,000 trajectory samples
* 100 time points per trajectory
* 2 variables per time point (prey and predator populations)

Data is converted to text format for language model processing with appropriate formatting and precision.

## Results
!!!! change
The LoRA fine-tuned model achieved:

* Overall MAE: 0.203 (72% improvement over baseline)
* Prey MAE: 0.217 (75% improvement over baseline)
* Predator MAE: 0.189 (68% improvement over baseline)
* Success Rate: 98% (49/50 trajectories successfully generated)
* Trainable Parameters: 242,688 (0.048% of the model's 0.5B parameters)

