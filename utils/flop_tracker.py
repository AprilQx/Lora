import numpy as np
from typing import Dict, Optional, Tuple, Union
import json
from pathlib import Path
import time

class FLOPTracker:
    """
    Utility class to track FLOPs used in Qwen2.5 model experiments.
    """
    
    def __init__(self, 
                 hidden_size: int = 896,
                 num_attention_heads: int = 14,
                 num_hidden_layers: int = 24,
                 intermediate_size: int = 4864,
                 head_dim: int = 64,
                 vocab_size: int = 151936,
                 max_budget: float = 1e17,
                 log_path: str = "flop_logs",
                 experiment_name: str = "default_experiment"):
        """
        Initialize the FLOP tracker with model parameters.
        
        Args:
            hidden_size: Model's hidden dimension
            num_attention_heads: Number of attention heads
            num_hidden_layers: Number of transformer layers
            intermediate_size: Dimension of intermediate MLP representations
            head_dim: Dimension of each attention head
            vocab_size: Size of vocabulary
            max_budget: Maximum FLOP budget (default: 10^17)
            log_path: Directory to save logs
            experiment_name: Name of the current experiment
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        
        self.max_budget = max_budget
        self.total_flops = 0
        self.experiment_log = []
        
        # Create log directory
        self.log_dir = Path(log_path)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"{experiment_name}_flop_log.json"
        
        # Initialize log file
        self._initialize_log()
    
    def _initialize_log(self):
        """Initialize the log file with experiment metadata."""
        metadata = {
            "experiment_name": self.log_file.stem.replace("_flop_log", ""),
            "model_config": {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_hidden_layers": self.num_hidden_layers,
                "intermediate_size": self.intermediate_size,
                "head_dim": self.head_dim,
                "vocab_size": self.vocab_size
            },
            "max_budget": self.max_budget,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "operations": []
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _update_log(self, operation_data):
        """Update the log file with a new operation."""
        try:
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
            
            log_data["operations"].append(operation_data)
            log_data["total_flops"] = self.total_flops
            log_data["budget_used_percent"] = (self.total_flops / self.max_budget) * 100
            
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"Error updating log file: {e}")
    
    def calculate_forward_pass_flops(self, seq_len: int, batch_size: int = 1, logits_to_keep: int=1) -> float:
        """
        Calculate FLOPs for a single forward pass through the model.
        
        Args:
            seq_len: Length of input sequence
            batch_size: Batch size
            
        Returns:
            Total FLOPs for forward pass
        """
        # Embedding layer
        embedding_flops = batch_size * seq_len * self.hidden_size  # Adding position embeddings
        
        # Per layer calculations
        layer_flops = 0
        
        # 1. Layer norms (2 per layer)
        rms_norm_flops = batch_size * seq_len * (4 * self.hidden_size + 12)
        layer_norm_flops = 2 * rms_norm_flops
        
        # 2. Self-attention
        # Query, Key, Value projections
        qkv_proj_flops = (
            batch_size * seq_len * self.hidden_size * (2 * self.hidden_size - 1) *3  # simplify the grouped attention to standard multi-head attention
        )
        
        # Applying RoPE
        rope_flops = batch_size * 3 * self.num_attention_heads * seq_len * self.head_dim +batch_size*seq_len * self.num_attention_heads * self.head_dim*0.5 #here we only negate half of the original length of x
        #kv caching -> 0

        #sliding window ->0

        # Attention matrix calculations
        attn_score_flops = batch_size * self.num_attention_heads * seq_len * seq_len * (2 * self.head_dim - 1)

        #Attention mask 
        attn_mask_flops = batch_size * self.num_attention_heads * seq_len * seq_len

        attn_softmax_flops = batch_size * self.num_attention_heads * seq_len * (12 * seq_len - 1)
        attn_weighted_sum_flops = batch_size * self.num_attention_heads * seq_len * self.head_dim * (2 * seq_len - 1)
        
        # Output projection
        output_proj_flops = batch_size * seq_len * self.hidden_size * (2 * self.hidden_size - 1)
        
        # Total attention flops
        attention_flops = qkv_proj_flops + rope_flops + attn_score_flops + attn_softmax_flops + attn_weighted_sum_flops + output_proj_flops+attn_mask_flops
        
        # 3. MLP block
        # Gate and Up projections
        gate_up_proj_flops = 2 * batch_size * seq_len * self.intermediate_size * (2 * self.hidden_size - 1)
        
        # SwiGLU activation
        swiglu_flops = batch_size * seq_len * self.intermediate_size * 14 #sigmoid  \approx 13 flops, multiplication \approx 1 flop
        
        # Element-wise multiplication
        elem_mult_flops = batch_size * seq_len * self.intermediate_size
        
        # Down projection
        down_proj_flops = batch_size * seq_len * self.hidden_size * (2 * self.intermediate_size - 1)
        
        # Total MLP flops
        mlp_flops = gate_up_proj_flops + swiglu_flops + elem_mult_flops + down_proj_flops
        
        # 4. Residual connections
        residual_flops = 2 * batch_size * seq_len * self.hidden_size
        
        # Total per layer
        layer_flops = layer_norm_flops + attention_flops + mlp_flops + residual_flops
        
        # All layers
        all_layers_flops = self.num_hidden_layers * layer_flops
        
        # Final layer norm
        final_norm_flops = batch_size * seq_len * (4 * self.hidden_size + 12)
        
        # LM head (for inference only)
        if logits_to_keep ==1:
            lm_head_flops = batch_size * 1 * self.hidden_size * self.vocab_size
        else:
            lm_head_flops = batch_size * seq_len * self.hidden_size * self.vocab_size
        
        # Total forward pass flops
        forward_flops = embedding_flops + all_layers_flops + final_norm_flops + lm_head_flops
        
        return forward_flops
    
    def calculate_steps(self, 
                   dataset_size: int, 
                   batch_size: int, 
                   num_epochs: int,
                   train_val_split: float = 0.8) -> Dict[str, int]:
        """
        Calculate the number of steps for training and validation.
        
        Args:
            dataset_size: Total number of samples in the dataset
            batch_size: Batch size used for training/validation
            num_epochs: Number of training epochs
            train_val_split: Ratio of training to total data (default: 0.8 for 80/20 split)
            
        Returns:
            Dictionary containing training_steps, validation_steps, and total_steps
        """
        # Calculate dataset splits
        train_size = int(dataset_size * train_val_split)
        val_size = dataset_size - train_size
        
        # Calculate steps per epoch (rounding up for partial batches)
        train_steps_per_epoch = (train_size + batch_size - 1) // batch_size
        val_steps_per_epoch = (val_size + batch_size - 1) // batch_size
        
        # Calculate total steps
        total_training_steps = train_steps_per_epoch * num_epochs
        total_validation_steps = val_steps_per_epoch * num_epochs  # Assuming validation after each epoch
        
        return {
            "training_steps_per_epoch": train_steps_per_epoch,
            "validation_steps_per_epoch": val_steps_per_epoch,
            "total_training_steps": total_training_steps,
            "total_validation_steps": total_validation_steps,
            "total_steps": total_training_steps + total_validation_steps
        }
    
    def calculate_generation_flops(self, 
                                  context_len: int, 
                                  gen_len: int, 
                                  batch_size: int = 1,
                                  use_sliding_window: bool = False,
                                  sliding_window_size: int = 32768
                                  ) -> float:
        """
        Calculate FLOPs for generation with the model.
        
        Args:
            context_len: Length of input context
            gen_len: Number of tokens to generate
            batch_size: Batch size
            use_sliding_window: Whether to use sliding window attention
            sliding_window_size: Size of sliding window
            
        Returns:
            Total FLOPs for generation
        """
        # 1. Initial forward pass for context
        context_flops = self.calculate_forward_pass_flops(context_len, batch_size)
        
        # 2. Per-token generation
        generation_flops = 0
        
        for i in range(1, gen_len + 1):
            current_seq_len = context_len + i - 1
            
            # Attention span length (affected by sliding window)
            if use_sliding_window and current_seq_len > sliding_window_size:
                 attention_span = sliding_window_size
            else:
                 attention_span = current_seq_len

             
            # Per layer calculations
            layer_flops = 0
            
            # Layer norms (2 per layer)
            rms_norm_flops = batch_size * (4 * self.hidden_size + 12)
            layer_norm_flops = 2 * rms_norm_flops
            
            # QKV projections (for single token)
            qkv_proj_flops = (
                batch_size * self.hidden_size * (2 * self.hidden_size - 1) *3 # Q projection K, V projections -> standard multi-head attention
            )
            rope_flops = batch_size * 3 * self.num_attention_heads * attention_span * self.head_dim +batch_size*attention_span * self.num_attention_heads * self.head_dim*0.5 #here we only negate half of the original length of x

            # Attention computation
            attn_score_flops = batch_size * self.num_attention_heads * attention_span * (2 * self.head_dim - 1)
            attn_softmax_flops = batch_size * self.num_attention_heads * (12 * attention_span - 1)
            attn_weighted_sum_flops = batch_size * self.num_attention_heads * self.head_dim * (2 * attention_span - 1)
            
            # Output projection
            output_proj_flops = batch_size * self.hidden_size * (2 * self.hidden_size - 1)
            
            # Total attention flops
            attention_flops = qkv_proj_flops + attn_score_flops + attn_softmax_flops + attn_weighted_sum_flops + output_proj_flops+rope_flops
            
            # MLP block
            gate_up_proj_flops = 2 * batch_size * self.intermediate_size * (2 * self.hidden_size - 1)
            swiglu_flops = batch_size * self.intermediate_size * 14
            elem_mult_flops = batch_size * self.intermediate_size
            down_proj_flops = batch_size * self.hidden_size * (2 * self.intermediate_size - 1)
            
            mlp_flops = gate_up_proj_flops + swiglu_flops + elem_mult_flops + down_proj_flops
            
            # Residual connections
            residual_flops = 2 * batch_size * self.hidden_size
            
            # Total per layer
            layer_flops = layer_norm_flops + attention_flops + mlp_flops + residual_flops
            
            # All layers
            all_layers_flops = self.num_hidden_layers * layer_flops
            
            # Final layer norm
            final_norm_flops = batch_size * (4 * self.hidden_size + 12)
            
            # LM head (for the final token)
            lm_head_flops = batch_size * self.hidden_size * self.vocab_size
            
            # Total for this token
            token_flops = all_layers_flops + final_norm_flops + lm_head_flops
            
            generation_flops += token_flops
        
        # 3. Total generation FLOPs
        total_flops = context_flops + generation_flops
        
        return total_flops
    
    def log_training_step(self, 
                          seq_len: int, 
                          batch_size: int, 
                          is_validation: bool = False,
                          description: str = "Training step",
                          num_steps: int = 1) -> float:
        """
        Log FLOPs for a training step (forward + backward).
        
        Args:
            seq_len: Sequence length
            batch_size: Batch size
            description: Description of the operation
            
        Returns:
            FLOPs used in this step


        """
        forward_flops = self.calculate_forward_pass_flops(seq_len, batch_size)

        if is_validation:
            step_flops = forward_flops * num_steps
            operation_type = "validation"
        else:
            # Backward pass is 2× forward pass
            step_flops = 3 * forward_flops * num_steps
            operation_type = "training"
    
        self.total_flops += step_flops
        
        # Log the operation
        operation_data = {
            "type": "training",
            "description": description,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "flops": step_flops,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_flops_so_far": self.total_flops,
            "budget_used_percent": (self.total_flops / self.max_budget) * 100
        }
        
        self.experiment_log.append(operation_data)
        self._update_log(operation_data)
        
        # Check if we've exceeded budget
        if self.total_flops > self.max_budget:
            print(f"WARNING: FLOP budget exceeded! Used: {self.total_flops:.2e}, Budget: {self.max_budget:.2e}")
        
        return step_flops
    
    def log_inference(self, 
                     context_len: int, 
                     gen_len: int, 
                     batch_size: int = 1,
                     description: str = "Inference",
                     use_sliding_window: bool = False) -> float:
        """
        Log FLOPs for an inference/generation operation.
        
        Args:
            context_len: Context length
            gen_len: Number of tokens to generate
            batch_size: Batch size
            description: Description of the operation
            use_sliding_window: Whether sliding window attention is used
            
        Returns:
            FLOPs used in this operation
        """
        flops = self.calculate_generation_flops(
            context_len=context_len,
            gen_len=gen_len,
            batch_size=batch_size,
            use_sliding_window=use_sliding_window
        )
        
        self.total_flops += flops
        
        # Log the operation
        operation_data = {
            "type": "inference",
            "description": description,
            "context_len": context_len,
            "gen_len": gen_len,
            "batch_size": batch_size,
            "flops": flops,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_flops_so_far": self.total_flops,
            "budget_used_percent": (self.total_flops / self.max_budget) * 100
        }
        
        self.experiment_log.append(operation_data)
        self._update_log(operation_data)
        
        # Check if we've exceeded budget
        if self.total_flops > self.max_budget:
            print(f"WARNING: FLOP budget exceeded! Used: {self.total_flops:.2e}, Budget: {self.max_budget:.2e}")
        
        return flops
    
    def generate_report(self, file_path: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive report of FLOP usage.
        
        Args:
            file_path: Path to save the report (if None, returns without saving)
            
        Returns:
            Report dictionary
        """
        # Count operation types
        training_ops = [op for op in self.experiment_log if op["type"] == "training"]
        inference_ops = [op for op in self.experiment_log if op["type"] == "inference"]
        
        training_flops = sum(op["flops"] for op in training_ops)
        inference_flops = sum(op["flops"] for op in inference_ops)
        
        report = {
            "total_flops": self.total_flops,
            "budget_used_percent": (self.total_flops / self.max_budget) * 100,
            "training_flops": training_flops,
            "training_percent": (training_flops / self.total_flops) * 100 if self.total_flops > 0 else 0,
            "inference_flops": inference_flops,
            "inference_percent": (inference_flops / self.total_flops) * 100 if self.total_flops > 0 else 0,
            "operation_count": len(self.experiment_log),
            "training_operation_count": len(training_ops),
            "inference_operation_count": len(inference_ops),
            "budget_remaining": self.max_budget - self.total_flops
        }
        
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def plot_usage(self, file_path: Optional[str] = None):
        """
        Plot FLOP usage over time.
        
        Args:
            file_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Extract data for plotting
            operations = list(range(len(self.experiment_log)))
            cumulative_flops = [op["total_flops_so_far"] for op in self.experiment_log]
            operation_types = [op["type"] for op in self.experiment_log]
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot cumulative FLOPs
            plt.plot(operations, cumulative_flops, 'b-', linewidth=2)
            
            # Mark different operation types
            training_indices = [i for i, type in enumerate(operation_types) if type == "training"]
            inference_indices = [i for i, type in enumerate(operation_types) if type == "inference"]
            
            plt.scatter([operations[i] for i in training_indices], 
                       [cumulative_flops[i] for i in training_indices],
                       c='green', marker='o', label='Training')
            
            plt.scatter([operations[i] for i in inference_indices], 
                       [cumulative_flops[i] for i in inference_indices],
                       c='red', marker='x', label='Inference')
            
            # Draw budget line
            plt.axhline(y=self.max_budget, color='r', linestyle='--', label='FLOP Budget')
            
            # Formatting
            plt.title('Cumulative FLOP Usage')
            plt.xlabel('Operation')
            plt.ylabel('FLOPs')
            plt.yscale('log')
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            
            # Save or show
            if file_path:
                plt.savefig(file_path)
                plt.close()
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available. Install it to plot FLOP usage.")