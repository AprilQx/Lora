from utils.flop_tracker import FLOPTracker
import numpy as np
from typing import Dict, Optional, Tuple, Union
import json
from pathlib import Path
import time

class LoRAFLOPTracker(FLOPTracker):
    """
    Extended FLOPTracker that accounts for LoRA's reduced computation.
    """
    def __init__(self, 
                 hidden_size=896,
                 num_attention_heads=14,
                 num_hidden_layers=24,
                 intermediate_size=4864,
                 head_dim=64,
                 vocab_size=151936,
                 max_budget=1e17,
                 log_path="flop_logs",
                 experiment_name="default_experiment",
                 lora_r=8,
                 lora_target_modules=None):
        """
        Initialize the LoRA FLOP tracker with model parameters.
        
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
            lora_r: LoRA rank
            lora_target_modules: List of modules to apply LoRA to (default: ["q_proj", "v_proj"])
        """
        # Call parent constructor
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            vocab_size=vocab_size,
            max_budget=max_budget,
            log_path=log_path,
            experiment_name=experiment_name
        )
        
        self.lora_r = lora_r
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]

        # Track LoRA-specific FLOP usage
        self.lora_training_flops = 0
        self.lora_inference_flops = 0
        
        # Add LoRA info to log
        self._update_lora_metadata()
    
    def _update_lora_metadata(self):
        """Update the log file with LoRA metadata."""
        try:
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
            
            # Add LoRA-specific info
            log_data["model_config"]["lora_r"] = self.lora_r
            log_data["model_config"]["lora_target_modules"] = self.lora_target_modules
            
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"Error updating log file with LoRA metadata: {e}")
    
    def calculate_forward_pass_flops(self, seq_len, batch_size=1, logits_to_keep=1):
        """
        Calculate FLOPs for a single forward pass through the model with LoRA.
        
        Args:
            seq_len: Length of input sequence
            batch_size: Batch size
            logits_to_keep: Number of logits to compute (1 for inference, seq_len for training)
            
        Returns:
            Total FLOPs for forward pass
        """
        # Embedding layer (unchanged)
        embedding_flops = batch_size * seq_len * self.hidden_size
        
        # Per layer calculations
        layer_flops = 0
        
        # 1. Layer norms (2 per layer) (unchanged)
        rms_norm_flops = batch_size * seq_len * (4 * self.hidden_size + 12)
        layer_norm_flops = 2 * rms_norm_flops
        
        
        # 2. Self-attention
        # Count LoRA modules
        lora_q_proj = "q_proj" in self.lora_target_modules
        lora_k_proj = "k_proj" in self.lora_target_modules
        lora_v_proj = "v_proj" in self.lora_target_modules
        lora_o_proj = "o_proj" in self.lora_target_modules
        
        # Calculate QKV projection FLOPs with LoRA where applicable
        qkv_proj_flops = 0
        
        # Query projection
        if lora_q_proj:
            # LoRA: x @ A^T @ B^T
            # Step 1: x @ A^T (batch_size * seq_len * hidden_size) × (hidden_size × lora_r)
            q_lora_step1_flops = batch_size * seq_len * self.lora_r * (2 * self.hidden_size - 1)
            # Step 2: result @ B^T (batch_size * seq_len * lora_r) × (lora_r × hidden_size)
            q_lora_step2_flops = batch_size * seq_len * self.hidden_size * (2 * self.lora_r - 1)
            # Original projection still happens
            q_orig_flops = batch_size * seq_len * self.hidden_size * (2 * self.hidden_size - 1)
            
            q_proj_flops = q_orig_flops + q_lora_step1_flops + q_lora_step2_flops
        else:
            q_proj_flops = batch_size * seq_len * self.hidden_size * (2 * self.hidden_size - 1)
        
        # Key projection
        if lora_k_proj:
            k_lora_step1_flops = batch_size * seq_len * self.lora_r * (2 * self.hidden_size - 1)
            k_lora_step2_flops = batch_size * seq_len * self.hidden_size * (2 * self.lora_r - 1)
            k_orig_flops = batch_size * seq_len * self.hidden_size * (2 * self.hidden_size - 1)
            
            k_proj_flops = k_orig_flops + k_lora_step1_flops + k_lora_step2_flops
        else:
            k_proj_flops = batch_size * seq_len * self.hidden_size * (2 * self.hidden_size - 1)
        
        # Value projection
        if lora_v_proj:
            v_lora_step1_flops = batch_size * seq_len * self.lora_r * (2 * self.hidden_size - 1)
            v_lora_step2_flops = batch_size * seq_len * self.hidden_size * (2 * self.lora_r - 1)
            v_orig_flops = batch_size * seq_len * self.hidden_size * (2 * self.hidden_size - 1)
            
            v_proj_flops = v_orig_flops + v_lora_step1_flops + v_lora_step2_flops
        else:
            v_proj_flops = batch_size * seq_len * self.hidden_size * (2 * self.hidden_size - 1)
        
        qkv_proj_flops = q_proj_flops + k_proj_flops + v_proj_flops
        
        # Rest of the attention mechanism (unchanged)
        rope_flops = batch_size * 3 * self.num_attention_heads * seq_len * self.head_dim + batch_size * seq_len * self.num_attention_heads * self.head_dim * 0.5
        attn_score_flops = batch_size * self.num_attention_heads * seq_len * seq_len * (2 * self.head_dim - 1)
        attn_mask_flops = batch_size * self.num_attention_heads * seq_len * seq_len
        attn_softmax_flops = batch_size * self.num_attention_heads * seq_len * (12 * seq_len - 1)
        attn_weighted_sum_flops = batch_size * self.num_attention_heads * seq_len * self.head_dim * (2 * seq_len - 1)
        
        # Output projection
        if lora_o_proj:
            o_lora_step1_flops = batch_size * seq_len * self.lora_r * (2 * self.hidden_size - 1)
            o_lora_step2_flops = batch_size * seq_len * self.hidden_size * (2 * self.lora_r - 1)
            o_orig_flops = batch_size * seq_len * self.hidden_size * (2 * self.hidden_size - 1)
            
            output_proj_flops = o_orig_flops + o_lora_step1_flops + o_lora_step2_flops
        else:
            output_proj_flops = batch_size * seq_len * self.hidden_size * (2 * self.hidden_size - 1)
        
        # Total attention flops
        attention_flops = qkv_proj_flops + rope_flops + attn_score_flops + attn_softmax_flops + attn_weighted_sum_flops + output_proj_flops + attn_mask_flops
        
        # 3. MLP block (unchanged)
        gate_up_proj_flops = 2 * batch_size * seq_len * self.intermediate_size * (2 * self.hidden_size - 1)
        swiglu_flops = batch_size * seq_len * self.intermediate_size * 14
        elem_mult_flops = batch_size * seq_len * self.intermediate_size
        down_proj_flops = batch_size * seq_len * self.hidden_size * (2 * self.intermediate_size - 1)
        
        mlp_flops = gate_up_proj_flops + swiglu_flops + elem_mult_flops + down_proj_flops
        
        # 4. Residual connections (unchanged)
        residual_flops = 2 * batch_size * seq_len * self.hidden_size
        
        # Total per layer
        layer_flops = layer_norm_flops + attention_flops + mlp_flops + residual_flops
        
        # All layers
        all_layers_flops = self.num_hidden_layers * layer_flops
        
        # Final layer norm (unchanged)
        final_norm_flops = batch_size * seq_len * (4 * self.hidden_size + 12)
        
        # LM head (for inference only)
        if logits_to_keep == 1:
            lm_head_flops = batch_size * 1 * self.hidden_size * self.vocab_size
        else:
            lm_head_flops = batch_size * seq_len * self.hidden_size * self.vocab_size
        
        # Total forward pass flops
        forward_flops = embedding_flops + all_layers_flops + final_norm_flops + lm_head_flops
        
        return forward_flops
    


    
    def generate_report(self, file_path=None):
        """
        Generate a comprehensive report of FLOP usage with LoRA-specific information.
        
        Args:
            file_path: Path to save the report (if None, returns without saving)
            
        Returns:
            Report dictionary
        """
        # Get base report
        report = super().generate_report(None)  # Don't save yet, we'll add more info
        
        # Add LoRA-specific information
        report["lora_config"] = {
            "lora_r": self.lora_r,
            "lora_target_modules": self.lora_target_modules,
        }
        
        # Calculate parameter efficiency
        base_param_count = self.hidden_size * self.hidden_size * 4 * self.num_hidden_layers  # Approximate
        lora_param_count = len(self.lora_target_modules) * self.lora_r * (2 * self.hidden_size) * self.num_hidden_layers
        
        report["parameter_efficiency"] = {
            "base_model_params": base_param_count,
            "lora_params": lora_param_count,
            "trainable_param_percent": (lora_param_count / base_param_count) * 100
        }
        
        # Save if path provided
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def log_training_step(self, 
                          seq_len: int, 
                          batch_size: int, 
                          is_validation: bool = False,
                          is_test: bool = False,
                          description: str = "Training step",
                          num_steps: int = 1) -> float:
        """
        Log FLOPs for a training step (forward + backward) with LoRA.
        
        Args:
            seq_len: Sequence length
            batch_size: Batch size
            is_validation: Whether this is validation (no backward pass)
            is_test: Whether this is test (no backward pass)
            description: Description of the operation
            num_steps: Number of steps
            
        Returns:
            FLOPs used in this step
        """
        forward_flops = self.calculate_forward_pass_flops(seq_len, batch_size, seq_len)
        
        if is_validation or is_test:
            # Validation/test only has forward pass
            step_flops = forward_flops * num_steps
            operation_type = "validation" if is_validation else "test"
            self.lora_inference_flops += step_flops
        else:
            # Training has both forward and backward passes
            backward_flops = 2*forward_flops
            step_flops = (forward_flops + backward_flops) * num_steps
            operation_type = "training"
            self.lora_training_flops += step_flops
        
        self.total_flops += step_flops
        
        # Log the operation
        operation_data = {
            "type": operation_type,
            "description": description,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "forward_flops": forward_flops * num_steps,
            "backward_flops": (step_flops - forward_flops * num_steps) if not (is_validation or is_test) else 0,
            "total_flops": step_flops,
            "lora_r": self.lora_r,
            "lora_target_modules": self.lora_target_modules,
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
    def calculate_steps(self, 
                  dataset_size: int, 
                  batch_size: int, 
                  num_epochs: int,
                  train_ratio: float = 0.7, 
                  test_ratio: float = 0.15, 
                  val_ratio: float = 0.15,
                  avg_seq_len: int = 512) -> Dict[str, int]:
        """
        Calculate the number of steps for training, validation, and testing.
        Also estimates total FLOP usage based on LoRA configuration.
        
        Args:
            dataset_size: Total number of samples in the dataset
            batch_size: Batch size used for training/validation/testing
            num_epochs: Number of training epochs
            train_ratio: Ratio of training data (default: 0.7)
            test_ratio: Ratio of test data (default: 0.15)
            val_ratio: Ratio of validation data (default: 0.15)
            avg_seq_len: Average sequence length for FLOP estimation
            
        Returns:
            Dictionary containing training_steps, validation_steps, test_steps, total_steps, and estimated FLOPs
        """
        
        # Calculate dataset splits
        train_size = int(dataset_size * train_ratio)
        val_size = int(dataset_size * val_ratio)
        test_size = dataset_size - train_size - val_size  # Ensure all samples are accounted for
        
        # Calculate steps per epoch (rounding up for partial batches)
        train_steps_per_epoch = (train_size + batch_size - 1) // batch_size
        val_steps_per_epoch = (val_size + batch_size - 1) // batch_size
        test_steps = (test_size + batch_size - 1) // batch_size
        
        # Calculate total training and validation steps
        total_training_steps = train_steps_per_epoch * num_epochs
        total_validation_steps = val_steps_per_epoch * num_epochs  # Assuming validation after each epoch
        
        # Store basic step information
        steps_info = {
            "training_steps_per_epoch": train_steps_per_epoch,
            "validation_steps_per_epoch": val_steps_per_epoch,
            "test_steps": test_steps,
            "total_training_steps": total_training_steps,
            "total_validation_steps": total_validation_steps,
            "total_steps": total_training_steps + total_validation_steps + test_steps
        }
        
        # Estimate FLOPs for different operations
        # Training steps (forward + backward)
        estimated_train_forward_flops = self.calculate_forward_pass_flops(avg_seq_len, batch_size, avg_seq_len) * total_training_steps
        estimated_train_backward_flops = 2*estimated_train_forward_flops  # Backward pass is twice as
        # Validation steps (forward only)
        estimated_val_flops = self.calculate_forward_pass_flops(avg_seq_len, batch_size, avg_seq_len) * total_validation_steps
        
        # Test steps (forward only)
        estimated_test_flops = self.calculate_forward_pass_flops(avg_seq_len, batch_size, avg_seq_len) * test_steps
        
        # Total estimated FLOPs
        total_estimated_flops = estimated_train_forward_flops + estimated_train_backward_flops + estimated_val_flops + estimated_test_flops
        
        # Add FLOP estimates to the result
        steps_info["estimated_flops"] = {
            "training_forward_flops": estimated_train_forward_flops,
            "training_backward_flops": estimated_train_backward_flops,
            "validation_flops": estimated_val_flops,
            "test_flops": estimated_test_flops,
            "total_flops": total_estimated_flops,
            "budget_used_percent": (total_estimated_flops / self.max_budget) * 100
        }
        
        # Add dataset split information
        steps_info["dataset_splits"] = {
            "total_samples": dataset_size,
            "train_samples": train_size,
            "validation_samples": val_size,
            "test_samples": test_size,
            "train_ratio": train_ratio,
            "validation_ratio": val_ratio,
            "test_ratio": test_ratio
        }
        
        return steps_info
    
    def calculate_generation_flops(self, 
                                  context_len: int, 
                                  gen_len: int, 
                                  batch_size: int = 1,
                                  use_sliding_window: bool = False,
                                  sliding_window_size: int = 32768) -> float:
        """
        Calculate FLOPs for generation with the model using LoRA.
        
        Args:
            context_len: Length of input context
            gen_len: Number of tokens to generate
            batch_size: Batch size
            use_sliding_window: Whether to use sliding window attention
            sliding_window_size: Size of sliding window
            
        Returns:
            Total FLOPs for generation
        """
        # For generation, LoRA doesn't change the computation pattern significantly
        # We just modify the appropriate methods to account for LoRA in the projections
        
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
            
            # Count LoRA modules
            lora_q_proj = "q_proj" in self.lora_target_modules
            lora_k_proj = "k_proj" in self.lora_target_modules
            lora_v_proj = "v_proj" in self.lora_target_modules
            
            # QKV projections with LoRA for single token
            q_proj_flops = 0
            if lora_q_proj:
                q_lora_step1_flops = batch_size * self.lora_r * (2 * self.hidden_size - 1)
                q_lora_step2_flops = batch_size * self.hidden_size * (2 * self.lora_r - 1)
                q_orig_flops = batch_size * self.hidden_size * (2 * self.hidden_size - 1)
                q_proj_flops = q_orig_flops + q_lora_step1_flops + q_lora_step2_flops
            else:
                q_proj_flops = batch_size * self.hidden_size * (2 * self.hidden_size - 1)
            
            k_proj_flops = 0
            if lora_k_proj:
                k_lora_step1_flops = batch_size * self.lora_r * (2 * self.hidden_size - 1)
                k_lora_step2_flops = batch_size * self.hidden_size * (2 * self.lora_r - 1)
                k_orig_flops = batch_size * self.hidden_size * (2 * self.hidden_size - 1)
                k_proj_flops = k_orig_flops + k_lora_step1_flops + k_lora_step2_flops
            else:
                k_proj_flops = batch_size * self.hidden_size * (2 * self.hidden_size - 1)
            
            v_proj_flops = 0
            if lora_v_proj:
                v_lora_step1_flops = batch_size * self.lora_r * (2 * self.hidden_size - 1)
                v_lora_step2_flops = batch_size * self.hidden_size * (2 * self.lora_r - 1)
                v_orig_flops = batch_size * self.hidden_size * (2 * self.hidden_size - 1)
                v_proj_flops = v_orig_flops + v_lora_step1_flops + v_lora_step2_flops
            else:
                v_proj_flops = batch_size * self.hidden_size * (2 * self.hidden_size - 1)
            
            qkv_proj_flops = q_proj_flops + k_proj_flops + v_proj_flops
            
            # RoPE
            rope_flops = batch_size * 3 * self.num_attention_heads * attention_span * self.head_dim + batch_size * attention_span * self.num_attention_heads * self.head_dim * 0.5
            
            # Attention computation
            attn_score_flops = batch_size * self.num_attention_heads * attention_span * (2 * self.head_dim - 1)
            attn_softmax_flops = batch_size * self.num_attention_heads * (12 * attention_span - 1)
            attn_weighted_sum_flops = batch_size * self.num_attention_heads * self.head_dim * (2 * attention_span - 1)
            
            # Output projection
            lora_o_proj = "o_proj" in self.lora_target_modules
            if lora_o_proj:
                o_lora_step1_flops = batch_size * self.lora_r * (2 * self.hidden_size - 1)
                o_lora_step2_flops = batch_size * self.hidden_size * (2 * self.lora_r - 1)
                o_orig_flops = batch_size * self.hidden_size * (2 * self.hidden_size - 1)
                output_proj_flops = o_orig_flops + o_lora_step1_flops + o_lora_step2_flops
            else:
                output_proj_flops = batch_size * self.hidden_size * (2 * self.hidden_size - 1)
            
            # Total attention flops
            attention_flops = qkv_proj_flops + rope_flops + attn_score_flops + attn_softmax_flops + attn_weighted_sum_flops + output_proj_flops
            
            # MLP block (unchanged)
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
    def log_inference(self, 
                     context_len: int, 
                     gen_len: int, 
                     batch_size: int = 1,
                     description: str = "Inference",
                     use_sliding_window: bool = False) -> float:
        """
        Log FLOPs for an inference/generation operation with LoRA.
        
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
        self.lora_inference_flops += flops
        
        # Log the operation
        operation_data = {
            "type": "inference",
            "description": description,
            "context_len": context_len,
            "gen_len": gen_len,
            "batch_size": batch_size,
            "flops": flops,
            "lora_r": self.lora_r,
            "lora_target_modules": self.lora_target_modules,
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
        Generate a comprehensive report of FLOP usage with LoRA-specific information.
        
        Args:
            file_path: Path to save the report (if None, returns without saving)
            
        Returns:
            Report dictionary
        """
        # Get base report
        report = super().generate_report(None)  # Don't save yet, we'll add more info
        
        # Add LoRA-specific information
        report["lora_config"] = {
            "lora_r": self.lora_r,
            "lora_target_modules": self.lora_target_modules,
        }
        
        # Calculate parameter efficiency
        base_param_count = self.hidden_size * self.hidden_size * 4 * self.num_hidden_layers  # Approximate
        lora_param_count = len(self.lora_target_modules) * self.lora_r * (2 * self.hidden_size) * self.num_hidden_layers
        
        report["parameter_efficiency"] = {
            "base_model_params": base_param_count,
            "lora_params": lora_param_count,
            "trainable_param_percent": (lora_param_count / base_param_count) * 100
        }
        
        # Add LoRA-specific FLOP breakdown
        report["lora_flops"] = {
            "lora_training_flops": self.lora_training_flops,
            "lora_inference_flops": self.lora_inference_flops,
            "lora_total_flops": self.lora_training_flops + self.lora_inference_flops
        }
        
        # Save if path provided
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def plot_usage(self, save_path: Optional[Union[str, Path]] = None):
        """
        Plot the FLOP usage over time with LoRA-specific information.
        
        Args:
            save_path: Path to save the plot (if None, shows the plot)
        """
        # Get base plot
        super().plot_usage(None)