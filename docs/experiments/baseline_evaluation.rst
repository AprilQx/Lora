Baseline Evaluation
=================

Overview
--------

We evaluated the zero-shot forecasting capabilities of the base Qwen2.5-0.5B-Instruct model without any fine-tuning. This establishes a performance baseline for comparison with our fine-tuned models.

Methodology
----------

The baseline evaluation:

1. Uses the pre-trained Qwen2.5-0.5B-Instruct model with frozen parameters
2. Takes the first 50 timesteps of each trajectory as input
3. Generates predictions for the next 50 timesteps
4. Compares predictions against ground truth values using MAE, MSE, and other metrics

Implementation
------------

.. code-block:: python

    def main(args):
        # Load model and tokenizer
        model, tokenizer = load_qwen()
        
        # Setup device
        device = setup_device(args.force_device)
        model = model.to(device)
        
        # Create FLOP tracker
        flop_tracker = FLOPTracker(
            max_budget=args.max_flops,
            experiment_name="baseline_evaluation",
            log_path=str(RESULTS_DIR / "flop_logs"),
            hidden_size=896,  # Qwen2.5-0.5B parameters
            num_attention_heads=14,
            num_hidden_layers=24,
            intermediate_size=4864,
            head_dim=64,
            vocab_size=151936
        )
        
        # Evaluation config
        config = {
            "input_steps": args.input_steps,
            "forecast_steps": args.forecast_steps,
            "alpha": args.alpha,
            "precision": args.precision,
            "max_tokens": 1000,
        }
        
        # Evaluate model
        results, successful_results, total_flops = evaluate_model_on_dataset(
            model=model,
            tokenizer=tokenizer,
            text_file_path=args.text_file_path if args.use_text_files else None,
            num_samples=args.num_samples,
            config=config,
            tracker=flop_tracker,
            visualize_first_n=args.visualize_first_n,
            precision=args.precision
        )

Results
------

The baseline Qwen2.5-0.5B-Instruct model achieved:

* **Overall MAE**: 0.723
* **Prey MAE**: 0.854
* **Predator MAE**: 0.592
* **Success Rate**: 78% (39/50 trajectories successfully generated)
* **FLOP Usage**: 3.42e13 FLOPs (0.0342% of budget)

.. figure:: ../../results/figures/baseline_error_distributions.png
   :alt: Baseline Error Distributions
   :width: 700px
   
   Distribution of prediction errors for the baseline model.