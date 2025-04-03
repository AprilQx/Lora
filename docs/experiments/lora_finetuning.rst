LoRA Fine-tuning
================

Overview
--------

We implemented Low-Rank Adaptation (LoRA) to fine-tune the Qwen2.5-0.5B-Instruct model for time series forecasting while minimizing the number of trainable parameters.

Training Methodology
------------------

Our fine-tuning process:

1. Freezes the base model parameters
2. Applies LoRA to attention query and value projection matrices
3. Makes the LM head bias trainable
4. Uses sliding window processing for handling long sequences
5. Employs AdamW optimizer with learning rate scheduling

Implementation Details
-------------------

.. code-block:: python

    def train_lora(
        model, tokenizer, train_texts, val_data_path,
        lora_r=8, lora_alpha=16, lora_dropout=0.05,
        learning_rate=1e-4, batch_size=4, max_steps=10000,
        max_length=512, eval_steps=500, save_steps=500,
        output_dir="results/models/lora", flop_tracker=None,
        use_wandb=False, wandb_project=None, wandb_entity=None,
        random_seed=42, precision=2
    ):
        # Apply LoRA to model
        model, trainable_params = apply_lora_to_model(
            model, 
            r=lora_r, 
            alpha=lora_alpha, 
            dropout=lora_dropout
        )
        
        # Prepare training data
        train_dataset = prepare_dataset(train_texts, tokenizer, max_length)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Train model
        for step in range(max_steps):
            # Training step
            model.train()
            batch = get_batch(train_dataset, batch_size)
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Evaluate periodically
            if step % eval_steps == 0:
                val_metrics = evaluate(model, tokenizer, val_data_path)
                
                # Log metrics
                if use_wandb:
                    wandb.log({"val/mae": val_metrics["mae"], "step": step})
                    
            # Save checkpoint
            if step % save_steps == 0:
                save_lora_model(model, f"{output_dir}/checkpoint_{step}")
                
        return model, history

Results
------

The LoRA fine-tuned model achieved:

* **Overall MAE**: 0.203 (**72% improvement** over baseline)
* **Prey MAE**: 0.217 (**75% improvement** over baseline)
* **Predator MAE**: 0.189 (**68% improvement** over baseline)
* **Success Rate**: 98% (49/50 trajectories successfully generated)
* **Trainable Parameters**: 242,688 (0.048% of the model's 0.5B parameters)

.. figure:: ../../results/figures/lora_vs_baseline.png
   :alt: LoRA vs Baseline Performance
   :width: 700px
   
   Comparison of prediction accuracy between baseline and LoRA fine-tuned model.