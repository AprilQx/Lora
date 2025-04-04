Reproducibility
==============

Hardware and Computational Requirements
--------------------------------------

### Hardware Specifications
We conducted our experiments using the following hardware:

* **Training**: NVIDIA A100 GPU with 40GB VRAM (colab/CSD3)
* **Inference**: Apple M1 Pro chip with 16GB unified memory
* **Storage**: Minimum 50GB for datasets, model checkpoints, and results
  
### Computational Time

.. list-table:: Experiment Timing
   :header-rows: 1
   :widths: 40 30 30

   * - Experiment
     - Hardware
     - Approx Time
   * - Hyperparameter Search
     - A100 GPU
     - 4.5 hours
   * - Context Length Study
     - A100 GPU
     - 1 hour
   * - Final Model Training
     - A100 GPU
     - 50 minutes

Our experiments were designed to be computationally efficient, enabling the entire pipeline to be run on a single GPU within a reasonable timeframe. The hyperparameter search phase was the most computationally intensive, but with our efficient FLOP tracking implementation, we were able to explore a wide range of configurations while staying well within our computational budget.

Step-by-Step Commands
-------------------

1. **Data Processing**

   Process the raw data with different precision settings:

   .. code-block:: bash

      # Process with 2 decimal places
      python src/data/preprocess_data.py --precision 2 --alpha 10.0 --output_dir data/processed2

      # Process with 3 decimal places
      python src/data/preprocess_data.py --precision 3 --alpha 10.0 --output_dir data/processed3

   **Parameters explained**:
   
   * ``--precision``: Number of decimal places (2 or 3)
   * ``--alpha``: Scaling factor for numerical values (10.0)
   * ``--output_dir``: Directory to save processed data

2. **Baseline Model Evaluation**:

   Evaluate the zero-shot performance of the untrained Qwen2.5 model:

   .. code-block:: bash

      # Evaluate baseline with precision 2
      python experiments/eval/base_eval_v3.py \
        --use_text_files \
        --text_file_path data/processed2/test_texts.txt \
        --num_samples 50 \
        --precision 2 \
        --visualize_first_n 5

      # Evaluate baseline with precision 3
      python experiments/eval/base_eval_v2.py \
        --use_text_files \
        --text_file_path data/processed3/test_texts.txt \
        --num_samples 50 \
        --precision 3 \
        --visualize_first_n 5

   **Parameters explained**:
   
   * ``--use_text_files``: Use preprocessed text files instead of raw HDF5 data
   * ``--text_file_path``: Path to the test data file
   * ``--num_samples``: Number of test trajectories to evaluate (50)
   * ``--precision``: Decimal precision to use (2 or 3)
   * ``--visualize_first_n``: Number of predictions to visualize (5)

   .. note::
      If you would like to change the eval for different precision, remember to change the precision accordingly for *Data_Dir* in the *src/eval/visualization.py*

3. **Hyperparameter Search**:

   Run a grid search over learning rates, LoRA ranks, and precision values:

   .. code-block:: bash

      python experiments/hyperparameter/hyper.py --max_flops 1e17 --use_wandb

   **Parameters explained**:
   
   * ``--max_flops``: Maximum FLOP budget (10^17)
   * ``--use_wandb``: Enable Weights & Biases tracking for visualization

   Each configuration trains for up to 2,000 steps. Results are saved to results/hyperparameter_search/ and tracked in Weights & Biases.

4. **Context Length Study**:

   .. code-block:: bash

      python experiments/context/context.py \
      --model_1 \
      --lr 1e-4 \
      --rank 8 \
      --precision 2

   **Parameters explained**:
   
   * ``--model_1``: Use the best model from hyperparameter search
   * ``--lr``: Learning rate (1e-4)
   * ``--rank``: LoRA rank (8)
   * ``--precision``: Precision value (2)

   This script evaluates context lengths of 128, 512, and 768 tokens, starting from the best model's weights. Results are saved to results/context_length_finetuning/.

5. **Final Model Training**:

   Train the final model with optimal hyperparameters:

   .. code-block:: bash

      python experiments/finetune/finetune.py \
        --train_file data/processed2/train_texts.txt \
        --val_file data/processed2/val_texts.txt \
        --output_dir results/final-finetune/models \
        --lora_r 8 \
        --lora_alpha 16 \
        --learning_rate 1e-4 \
        --batch_size 4 \
        --max_steps 15000 \
        --wandb_project lora-finetuning_final

6. **Final Evaluation**:

   Evaluate the fine-tuned model on the test set:

   .. code-block:: bash

      python experiments/eval/finetune_eval.py \
        --use_text_files \
        --visualize_first_n 20