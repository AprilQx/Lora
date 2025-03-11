"""
Preprocessor for time series data based on the LLMTIME approach.
Handles loading, preprocessing, and converting between numerical arrays and text representations.
"""




import h5py
import numpy as np
from typing import Tuple, List, Union, Dict, Optional
import torch


def load_and_preprocess(
    file_path: str,
    tokenizer=None,
    val_split: float = 0.2,
    alpha: float = 10,
    precision: int = 3,
    seed: int = 42,
    max_length: int = 512,
    stride: int = 256,
) -> Tuple[List[str], List[str]]:
    """
    Load Lotka-Volterra data from HDF5 file and preprocess it into text format.
    
    Args:
        file_path: Path to the HDF5 file
        tokenizer: Tokenizer to use for tokenizing the text (optional)
        val_split: Fraction of data to use for validation
        alpha: Scaling parameter to normalize values
        precision: Number of decimal places to round to
        seed: Random seed for shuffling
        max_length: Maximum length of each chunk when tokenizing
        stride: Stride between consecutive chunks when tokenizing
        
    Returns:
        Tuple of (train_texts, val_texts) where each is a list of formatted text sequences
        If tokenizer is provided, returns tokenized sequences instead
    """
    # Load the data
    trajectories, time_points = load_data(file_path)
    
    # Split the data into training and validation sets
    np.random.seed(seed)
    indices = np.random.permutation(trajectories.shape[0])
    split_idx = int(trajectories.shape[0] * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_trajectories = trajectories[train_indices]
    val_trajectories = trajectories[val_indices]
    
    
    # Preprocess the data
    train_texts = []
    for i, trajectory in enumerate(train_trajectories):
        text = numeric_to_text(trajectory, alpha=alpha, precision=precision)
        train_texts.append(text)
       
    
    val_texts = []
    for i, trajectory in enumerate(val_trajectories):
        text = numeric_to_text(trajectory, alpha=alpha, precision=precision)
        val_texts.append(text)
    
    
    if tokenizer is not None:
        train_tokens = process_sequences(train_texts, tokenizer, max_length=max_length, stride=stride)
        val_tokens = process_sequences(val_texts, tokenizer, max_length=max_length, stride=stride)
        return train_tokens, val_tokens
    
    return train_texts, val_texts


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


def text_to_numeric(
    text: str
) -> np.ndarray:
    """
    Convert text representation back to numeric array and rescale to original range.
    
    Args:
        text: Text representation of time series
        scaling_factor: If provided, use this explicit scaling factor to rescale the data
    
    Returns:
        NumPy array of shape (time_steps, variables)
    """
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
        
        # Convert to float - handle potential errors more gracefully
        try:
            variables_numeric = [float(v) for v in variables]
            numeric_data.append(variables_numeric)
        except ValueError as e:
            print(f"Warning: Could not convert value in '{timestep}': {e}")
            continue
    
    # Convert to NumPy array
    if not numeric_data:
        return np.array([])
        
    numeric_array = np.array(numeric_data)
    
    # Return as is if no scaling information
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


def check_all_trajectories_distribution(file_path: str, alpha: float = 10.0):
    """
    Check the distribution statistics for all trajectories in the dataset.
    
    Args:
        file_path: Path to the HDF5 file
        alpha: Target scale for the 99th percentile
    
    Returns:
        Aggregated statistics
    """
    # Load the data
    trajectories, _ = load_data(file_path)
    
    # Initialize counters
    total_points = 0
    points_above_10 = 0
    points_above_20 = 0
    all_scaled_values = []
    
    # Process each trajectory
    for trajectory in trajectories:
        # Scale the data
        scaled_data = scale_data(trajectory, alpha)
        flat_data = scaled_data.flatten()
        
        # Update counters
        total_points += flat_data.size
        points_above_10 += np.sum(flat_data > 10.0)
        points_above_20 += np.sum(flat_data > 20.0)
        
        # Collect all scaled values for overall statistics
        all_scaled_values.extend(flat_data)
    
    # Convert to numpy array for statistics
    all_scaled_values = np.array(all_scaled_values)
    
    # Calculate statistics
    stats = {
        "min": float(np.min(all_scaled_values)),
        "max": float(np.max(all_scaled_values)),
        "mean": float(np.mean(all_scaled_values)),
        "median": float(np.median(all_scaled_values)),
        "p99": float(np.percentile(all_scaled_values, 99)),
        "p95": float(np.percentile(all_scaled_values, 95)),
        "percent_above_10": float((points_above_10 / total_points) * 100),
        "percent_above_20": float((points_above_20 / total_points) * 100),
        "num_trajectories": int(trajectories.shape[0]),
        "total_points": int(total_points)
    }
    
    return stats


if __name__ == "__main__":
    # Simple test to demonstrate usage
    import torch
    from qwen import load_qwen
    
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
        
        # Check distribution of all trajectories
        try:
            all_stats = check_all_trajectories_distribution(file_path, alpha)
            print(f"All trajectories stats: {all_stats}")
        except Exception as e:
            print(f"Error checking all trajectories: {e}")
    
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

