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
    alpha: float = 0.99,
    precision: int = 3,
    seed: int = 42,
    max_length: int = 512,
    stride: int = 256,
) -> Tuple[List[str], List[str]]:
    """
    Load Lotka-Volterra data from HDF5 file and preprocess it into text format.
    
    Args:
        file_path: Path to the HDF5 file
        val_split: Fraction of data to use for validation
        alpha: Scaling parameter to normalize values
        precision: Number of decimal places to round to
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (train_texts, val_texts) where each is a list of formatted text sequences
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
    for trajectory in train_trajectories:
        train_texts.append(numeric_to_text(trajectory, alpha=alpha, precision=precision))
    
    val_texts = []
    for trajectory in val_trajectories:
        val_texts.append(numeric_to_text(trajectory, alpha=alpha, precision=precision))
    
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


def numeric_to_text(
    trajectory: np.ndarray,
    alpha: float = 0.99,
    precision: int = 3,
) -> str:
    """
    Convert a numeric trajectory to text format following the LLMTIME scheme.
    
    Args:
        trajectory: NumPy array of shape (time_steps, variables)
        alpha: Scaling parameter (percentile used for scaling)
        precision: Number of decimal places to keep
    
    Returns:
        Formatted text string
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


def scale_data(
    data: np.ndarray,
    alpha: float = 0.99,
) -> np.ndarray:
    """
    Scale the data using a percentile-based approach.
    
    Args:
        data: NumPy array of data
        alpha: Percentile to use for scaling (0.99 means scale so 99th percentile equals 1)
    
    Returns:
        Scaled data
    """
    # Find the scaling factor based on the specified percentile
    percentile_value = np.percentile(data, alpha * 100)
    
    # Avoid division by zero
    if percentile_value == 0:
        print("Warning: 0 percentile value, skipping scaling")
        percentile_value = 1.0
    
    # Scale the data
    scale_data = data / percentile_value
    # Print statistics for validation
    return scale_data


def text_to_numeric(
    text: str,
    alpha: float = 0.99,
    precision: int = 3,
) -> np.ndarray:
    """
    Convert text representation back to numeric array.
    
    Args:
        text: Text representation of time series
        alpha: Scaling parameter used during encoding
        precision: Precision used in encoding
    
    Returns:
        NumPy array of shape (time_steps, variables)
    """
    # Split into timesteps
    timesteps = text.split(";")
    
    # Remove any empty timesteps
    timesteps = [ts for ts in timesteps if ts.strip()]
    
    # Initialize list to hold numeric values
    numeric_data = []
    
    # Determine the expected number of variables by looking at the first valid timestep
    if timesteps:
        expected_vars = len(timesteps[0].split(","))
    else:
        return np.array([])
    
    # Process each timestep
    for timestep in timesteps:
        # Split variables at this timestep
        variables = timestep.split(",")
        
        # Skip if we don't have the right number of variables
        if len(variables) != expected_vars:
            continue
        
        # Convert to float
        try:
            variables_numeric = [float(v) for v in variables]
            numeric_data.append(variables_numeric)
        except ValueError:
            # Handle potential parsing errors (e.g., 'NaN' or incomplete values)
            continue
    
    # Check if we have valid data
    if not numeric_data:
        return np.array([])
    
    # Convert to NumPy array
    return np.array(numeric_data)

def process_sequences(texts, tokenizer, max_length=512, stride=256):
    """
    Process text sequences into tokenized chunks for training.
    
    Args:
        texts: List of text sequences
        tokenizer: Tokenizer to use
        max_length: Maximum length of each chunk
        stride: Stride between consecutive chunks
        
    Returns:
        Tensor of tokenized input_ids
    """
    all_input_ids = []
    for text in texts:
        # Apply tokenization scheme to the text:
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]

        # Create sliding windows to further divide the data into chunks:
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]
            if len(chunk) < max_length:
                chunk = torch.cat(
                    [
                        chunk,
                        torch.full((max_length - len(chunk),), tokenizer.pad_token_id, 
                                   dtype=chunk.dtype, device=chunk.device),
                    ]
                )
            all_input_ids.append(chunk)
            
    if not all_input_ids:
        return torch.tensor([])
    
    return torch.stack(all_input_ids)

def example_tokenization(tokenizer, raw_text):
    """Display example of tokenization process"""
    # Show the raw text
    print("Raw text example:")
    print(raw_text[:100] + "...")
    
    # Show tokenized results
    tokens = tokenizer(raw_text[:100], return_tensors="pt", add_special_tokens=False)
    print("\nTokenized (first 100 chars):")
    print(f"Token IDs: {tokens.input_ids[0][:50]}...")
    
    # Decode back to verify
    decoded = tokenizer.decode(tokens.input_ids[0][:50])
    print(f"Decoded back: {decoded}...")
    
    return tokens.input_ids[0][:20].tolist()
 
if __name__ == "__main__":
    # Simple test to demonstrate usage
    import torch
    from qwen import load_qwen
    
    model, tokenizer = load_qwen()
    
    # Load test data
    train_texts, val_texts = load_and_preprocess("data/lotka_volterra_data.h5")
    
    # Print an example
    print("Example text representation:")
    print(train_texts[0][:100] + "...")
    
    # Convert back to numeric
    numeric_data = text_to_numeric(train_texts[0])
    print(f"\nConverted back to numeric, shape: {numeric_data.shape}")
    print(numeric_data[:3])
    
    if train_texts:
        token_ids = example_tokenization(tokenizer, train_texts[0])
        
        # Process sequences - this is what we'll use for training
        print("\nProcessing sequences...")
        processed = process_sequences([train_texts[0]], tokenizer, max_length=32, stride=16)
        print(f"Processed shape: {processed.shape}")
        print(f"First chunk: {processed[0][:20]}...")
   