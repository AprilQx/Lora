"""
Preprocessor for time series data based on the LLMTIME approach.
Handles loading, preprocessing, and converting between numerical arrays and text representations.
"""




import h5py
import numpy as np
from typing import Tuple, List, Union, Dict, Optional
import torch
import logging
import os
import json


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Tuple of (trajectories, time_points)
    """
    with h5py.File(file_path, "r") as f:
        trajectories = f["trajectories"][:]  # Shape: (1000, 100, 2)
        time_points = f["time"][:]          # Shape: (100,)
    
    return trajectories, time_points



def scale_data(
    data: np.ndarray,
    alpha: float = 10,
) -> np.ndarray:
    """
    Scale the data using the 99th percentile to fit within range 0-alpha.
    Each series (each column of data) is scaled independently.
    
    Args:
        data: NumPy array of data with shape (time_steps, variables)
        alpha: Target scale value - the 99th percentile will be scaled to this value
    
    Returns:
        Scaled data with the same shape as the input
    """
    # Create a copy to avoid modifying the original
    scaled_data = np.zeros_like(data, dtype=float)
    
    # Scale each variable/series independently
    for var_idx in range(data.shape[1]):
        series = data[:, var_idx]
        
        # Find the 99th percentile for this series
        p99 = np.percentile(series, 99)
        
        # Avoid division by zero or very small values
        if p99 <= 1e-10:
            print(f"Warning: Very small or zero 99th percentile for variable {var_idx}, using 1.0")
            p99 = 1.0
        
        # Scale this series so its 99th percentile equals alpha
        scaled_data[:, var_idx] = series * (alpha / p99)
    
    return scaled_data


def numeric_to_text(
    trajectory: np.ndarray,
    alpha: float = 10,
    precision: int = 3,
) -> Tuple[str, float]:
    """
    Convert a numeric trajectory to text format following the LLMTIME scheme.
    
    Args:
        trajectory: NumPy array of shape (time_steps, variables)
        alpha: Scaling parameter (max value after scaling will be alpha)
        precision: Number of decimal places to keep
    
    Returns:
        Tuple of (formatted_text, scaling_factor)
    """
    # Scale the data
    scaled_trajectory = scale_data(trajectory, alpha)
    
    # Round to specified precision
    scaled_trajectory = np.round(scaled_trajectory, precision)
    
    # Convert to text format
    text_parts = []
    for timestep in scaled_trajectory:
        # Join variables at this timestep with commas
        timestep_str = ",".join([f"{value:.{precision}f}" for value in timestep])
        text_parts.append(timestep_str)
    
    # Join timesteps with semicolons
    return ";".join(text_parts)


def text_to_numeric(text):
    """
    Convert text representation back to numeric array with robust error handling.
    
    Args:
        text: Text representation of time series
        
    Returns:
        NumPy array of shape (time_steps, variables)
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)

    
    # Split into timesteps
    timesteps = text.split(";")
    
    # Remove any empty timesteps
    timesteps = [ts for ts in timesteps if ts.strip()]
    
    if not timesteps:
        return np.array([])
    
    # Initialize list to hold numeric values
    numeric_data = []
    
    # Process each timestep
    for timestep in timesteps:
        # Split variables at this timestep
        variables = timestep.split(",")
        
        # Ensure we have exactly 2 variables (prey and predator)
        if len(variables) != 2:
            logger.warning(f"Skipping timestep with incorrect variable count: {timestep}")
            continue
        
        # Convert to float - handle potential errors more gracefully
        try:
            variables_numeric = [float(v) for v in variables]
            numeric_data.append(variables_numeric)
        except ValueError as e:
            logger.warning(f"Warning: Could not convert value in '{timestep}': {e}")
            continue
    
    # Convert to NumPy array
    if not numeric_data:
        return np.array([])
        
    numeric_array = np.array(numeric_data)
    
    # Ensure we have a 2D array with shape (timesteps, 2)
    if len(numeric_array.shape) != 2 or numeric_array.shape[1] != 2:
        logger.warning(f"Incorrect array shape: {numeric_array.shape}, expected (n,2)")
        if len(numeric_array.shape) == 1:
            # Try to reshape a flat array
            numeric_array = numeric_array.reshape(-1, 2)
    
    return numeric_array


def process_sequences(texts, tokenizer, max_length=3200, stride=256):
    """
    Process text sequences into tokenized chunks for training and return a list
    that can be easily converted to a PyTorch Dataset.
    
    Args:
        texts: List of text sequences
        tokenizer: Tokenizer to use
        max_length: Maximum length of each chunk
        stride: Stride between consecutive chunks
        
    Returns:
        List of dictionaries containing input_ids and attention_masks
    """
    processed_examples = []
    
    for text_idx, text in enumerate(texts):
        # Apply tokenization scheme to the text
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]
        
        # If sequence is shorter than max_length, pad it
        if len(seq_ids) <= max_length:
            input_ids = torch.cat([
                seq_ids,
                torch.full(
                    (max_length - len(seq_ids),), 
                    tokenizer.pad_token_id,
                    dtype=seq_ids.dtype
                )
            ])
            attention_mask = torch.cat([
                torch.ones(len(seq_ids), dtype=torch.long),
                torch.zeros(max_length - len(seq_ids), dtype=torch.long)
            ])
            
            processed_examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "original_idx": text_idx
            })
        else:
            # Create sliding windows for longer sequences
            for i in range(0, len(seq_ids) - max_length + 1, stride):
                chunk = seq_ids[i:i + max_length]
                processed_examples.append({
                    "input_ids": chunk,
                    "attention_mask": torch.ones(max_length, dtype=torch.long),
                    "original_idx": text_idx,
                    "chunk_start": i
                })
            
            # Handle the final chunk if it doesn't align with stride
            if i + max_length < len(seq_ids):
                final_chunk = seq_ids[-max_length:]
                processed_examples.append({
                    "input_ids": final_chunk,
                    "attention_mask": torch.ones(max_length, dtype=torch.long),
                    "original_idx": text_idx,
                    "chunk_start": len(seq_ids) - max_length
                })
    
    return processed_examples

def check_scaling_distribution(
    data: np.ndarray,
    alpha: float = 10.0,
) -> Dict[str, float]:
    """
    Check the distribution of scaled data to ensure it meets our requirements.
    
    Args:
        data: NumPy array of data
        alpha: Target scale for the 99th percentile
    
    Returns:
        Dictionary with statistics about the scaled data distribution
    """
    # Scale the data
    scaled_data = scale_data(data, alpha)
    
    # Flatten the data if it's multi-dimensional
    flat_data = scaled_data.flatten()
    
    # Calculate statistics
    stats = {
        "min": float(np.min(flat_data)),
        "max": float(np.max(flat_data)),
        "mean": float(np.mean(flat_data)),
        "median": float(np.median(flat_data)),
        "p99": float(np.percentile(flat_data, 99)),
        "p95": float(np.percentile(flat_data, 95)),
        "percent_above_10": float((flat_data > 10.0).mean() * 100),
        "percent_above_20": float((flat_data > 20.0).mean() * 100),
    }
    
    return stats


if __name__ == "__main__":
    # Simple test to demonstrate usage
    import torch
    import sys
    from pathlib import Path
    file_path= Path(__file__).resolve()
    sys.path.append(str(file_path.parent.parent.parent))
    from src.models.qwen import load_qwen
    
    model, tokenizer = load_qwen()
    
    # Load data
    file_path = "data/lotka_volterra_data.h5"

    # Check scaling distribution at different alpha values
    print("\nScaling distribution statistics:")
    for alpha in [5.0, 10.0, 15.0]:
        print(f"\nWith alpha = {alpha}:")
        
        # Load a trajectory for testing
        trajectories, _ = load_data(file_path)
        example_trajectory = trajectories[0]
        text= numeric_to_text(example_trajectory, alpha=alpha, precision=3)
        token= tokenizer(text, return_tensors="pt").input_ids[0]
        print(f"Token Shape: {token.shape}")
        
        # Check distribution of single trajectory
        stats = check_scaling_distribution(example_trajectory, alpha)
        print(f"Single trajectory stats: {stats}")

    #some examples before and after tokenization
    # Load the data
    file_path = "data/lotka_volterra_data.h5"
    trajectories, _ = load_data(file_path)  # Load the data
    example_trajectory = trajectories[0]  # Select an example trajectory
    text = numeric_to_text(example_trajectory[:2], alpha=10, precision=3)  # Convert to text
    print(f"Text representation: {text}")  # Print the text representation
    token = tokenizer(text, return_tensors="pt").input_ids[0]  # Tokenize the text
    print(f"Token shape: {token.shape}")  # Print the shape of the tokenized text
    # Print the tokenized text
    print(f"Tokenized text: {token}")  # Print the tokenized text
    # Print the text representation
    print(f"Text representation: {text}")  # Print the text representation
    # Convert back to numeric
    numeric_array = text_to_numeric(text)  # Convert back to numeric
    print(f"Numeric representation: {numeric_array}")  # Print the numeric representation
    # Check the scaling distribution
    stats = check_scaling_distribution(example_trajectory, alpha=10)
    print(f"Single trajectory stats: {stats}")  # Print the statistics
        
    
# since we want the most of the data in the range 0-10, we will choose alpha=10 here

#With alpha = 10.0:
#All trajectories stats: {'min': 4.8468449676875025e-05, 'max': 12.877876281738281, 'mean': 3.760530948638916, 'median': 3.122563362121582, 'p99': 10.000000047683717, 'p95': 9.070397233963009, 'percent_above_10': 1.0, 'percent_above_20': 0.0, 'num_trajectories': 1000, 'total_points': 200000}

#analyse tokens per trajectory
# Load the data 
# file_path = "data/lotka_volterra_data.h5"
# trajectories, _ = load_data(file_path)  


# # Process the data
# train_tokens, val_tokens = load_and_preprocess(file_path, tokenizer=tokenizer, alpha=10, precision=3, seed=42)

# # Analyze the token counts
# train_token_counts = [len(tokens) for tokens in train_tokens]
# val_token_counts = [len(tokens) for tokens in val_tokens]

# # Print some statistics
# print("Train tokens:")
# print(f"  Total tokens: {sum(train_token_counts)}")
# print(f"  Min tokens: {min(train_token_counts)}")
# print(f"  Max tokens: {max(train_token_counts)}")
# print(f"  Mean tokens: {np.mean(train_token_counts)}")
# print(f"Average tokens per timestep: {np.mean(train_token_counts) / 100}")

# Train tokens:
#   Total tokens: 2048000
#   Min tokens: 512
#   Max tokens: 512
#   Mean tokens: 512.0
# Average tokens per timestep: 5.12

