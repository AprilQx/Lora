Context Length Exploration
========================

Overview
--------

We studied how varying the input context length affects model performance by continuing fine-tuning our best-performing LoRA model with different context window sizes.

Experiment Design
---------------

Starting from our best hyperparameter configuration (LoRA rank 8, learning rate 1e-4, precision 2), we conducted additional fine-tuning with:

* **Context lengths**: 128, 512, 768 tokens
* **Training steps**: 1,000 per configuration
* **Evaluation**: Every 500 steps on the validation set

Implementation
------------

The context length experiments used the existing weights from our best model as a starting point:

.. code-block:: python

    def run_context_length_finetuning(pretrained_model_path, lora_r, learning_rate, precision):
        # Define context lengths to explore
        context_lengths = [128, 512, 768]
        
        # Load pretrained weights
        pretrained_weights_path = Path(pretrained_model_path) / "lora_weights.pt"
        
        # For each context length
        for context_length in context_lengths:
            # Load fresh model
            model, tokenizer = load_qwen()
            
            # Apply LoRA to model
            model, trainable_params = apply_lora_to_model(
                model, 
                r=lora_r, 
                alpha=lora_alpha, 
                dropout=lora_dropout
            )
            
            # Load pretrained LoRA weights
            model = load_lora_weights(model, pretrained_weights_path)
            
            # Continue training with this context length
            trained_model, history = train_lora(
                model=model,
                tokenizer=tokenizer,
                train_texts=train_texts,
                val_data_path=val_file,
                max_length=context_length,  # This is what we're testing
                # Other parameters...
            )

Results
------

* **Optimal Context Length**: 512 tokens
* **Performance by Context Length**:
  * 128 tokens: MAE 0.245
  * 512 tokens: MAE 0.198
  * 768 tokens: MAE 0.212

.. figure:: ../../results/context_length_finetuning/context_length_mae_Model1.png
   :alt: Effect of Context Length on MAE
   :width: 700px
   
   Impact of context length on validation MAE.

Analysis
-------

The 512-token context provided the optimal balance between having sufficient history for accurate predictions and maintaining focus on the most relevant information. While the 768-token context included more historical data, it didn't lead to better performance, suggesting that more distant timesteps have diminishing relevance for prediction.

Computational Impact
-----------------

..