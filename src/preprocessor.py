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
) -> Tuple[np.ndarray, float]:
    """
    Scale the data using a percentile-based approach.
    
    Args:
        data: NumPy array of data
        alpha: Percentile to use for scaling (we want to scale the data in the range 0-10)
    
    Returns:
        Tuple of (scaled_data, scaling_factor)
        scaling_factor can be used to convert back to the original scale
    """
    # Find the scaling factor based on the specified percentile
    scaling_factor = np.percentile(data, 99)
    
    # Avoid division by zero
    if scaling_factor <= 0:
        print("Warning: Zero or negative percentile value, using 1.0 for scaling")
        scaling_factor = 1.0
    
    # Scale the data
    scaled_data = data * (alpha / scaling_factor)
    
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
                        torch.full(
                            (max_length - len(chunk),), 
                            tokenizer.pad_token_id,
                            dtype=chunk.dtype, 
                            device=chunk.device
                        ),
                    ]
                )
            all_input_ids.append(chunk)
            
    if not all_input_ids:
        return torch.tensor([])
    
    return torch.stack(all_input_ids)


def example_tokenization(tokenizer, raw_text):
    """
    Display example of tokenization process
    
    Args:
        tokenizer: Tokenizer to use
        raw_text: Raw text to tokenize
        
    Returns:
        List of token IDs (first 20)
    """
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
    
    # Convert back to numeric (note: we don't have scaling factor in this example)
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