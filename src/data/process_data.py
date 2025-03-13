
import h5py
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
from preprocessor import load_data, scale_data, numeric_to_text, load_and_preprocess

def create_data_split(
    file_path: str,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Create train/validation/test splits for the dataset.
        
        Args:
            file_path: Path to the HDF5 file
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """

        
        # Load the data to get the number of trajectories
    trajectories, _ = load_data(file_path)
    num_trajectories = trajectories.shape[0]
    
    # Create random permutation
    np.random.seed(seed)
    indices = np.random.permutation(num_trajectories)
    
    # Calculate split indices
    train_end = int(num_trajectories * train_split)
    val_end = train_end + int(num_trajectories * val_split)
    
    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    logger.info(f"Created data split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test")

    return train_indices, val_indices, test_indices


def preprocess_dataset(
    file_path: str,
    alpha: float = 10.0,
    precision: int = 3,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load and preprocess the dataset, creating train/val/test splits.
    
    Args:
        file_path: Path to the HDF5 file
        alpha: Scaling parameter
        precision: Decimal precision
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        seed: Random seed
    
    Returns:
        Tuple of (train_texts, val_texts, test_texts)
    """
    # Create the data splits
    train_indices, val_indices, test_indices = create_data_split(
        file_path, train_split, val_split, test_split, seed
    )
    
    # Load the data
    trajectories, _ = load_data(file_path)
    
    # Process each split
    train_texts = [numeric_to_text(trajectories[idx], alpha, precision) for idx in train_indices]
    val_texts = [numeric_to_text(trajectories[idx], alpha, precision) for idx in val_indices]
    test_texts = [numeric_to_text(trajectories[idx], alpha, precision) for idx in test_indices]
    
    logger.info(f"Preprocessed dataset into {len(train_texts)} train, {len(val_texts)} validation, and {len(test_texts)} test examples")
    
    return train_texts, val_texts, test_texts

def process_sequences_for_training(
    texts: List[str],
    tokenizer,
    max_length: int = 512,
    stride: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process text sequences into tokenized chunks for training.
    
    Args:
        texts: List of text sequences
        tokenizer: Tokenizer to use
        max_length: Maximum length of each chunk
        stride: Stride between consecutive chunks
        
    Returns:
        Tuple of (input_ids, labels) tensors
    """
    all_input_ids = []
    all_labels = []
    
    for text in texts:
        # Apply tokenization scheme
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]
        
        # If sequence is too long, create chunks with sliding window
        if len(seq_ids) > max_length:
            # Create sliding windows
            for i in range(0, len(seq_ids) - max_length + 1, stride):
                chunk = seq_ids[i:i + max_length]
                all_input_ids.append(chunk)
                all_labels.append(chunk.clone())  # For autoregressive loss
        else:
            # Pad sequence if it's shorter than max_length
            if len(seq_ids) < max_length:
                padding = torch.full((max_length - len(seq_ids),), tokenizer.pad_token_id)
                input_ids = torch.cat([seq_ids, padding])
                
                # Create labels (set padding to -100 to ignore in loss)
                labels = torch.cat([seq_ids, torch.full((max_length - len(seq_ids),), -100)])
            else:
                input_ids = seq_ids
                labels = seq_ids.clone()
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
    
    return torch.stack(all_input_ids), torch.stack(all_labels)

def save_preprocessed_data(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    output_dir: str,
    metadata: Dict = None
):
    """
    Save preprocessed data to files.
    
    Args:
        train_texts: List of training text sequences
        val_texts: List of validation text sequences
        test_texts: List of test text sequences
        output_dir: Directory to save files
        metadata: Optional metadata to save
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text data
    with open(os.path.join(output_dir, "train_texts.txt"), "w") as f:
        for text in train_texts:
            f.write(text + "\n")
    
    with open(os.path.join(output_dir, "val_texts.txt"), "w") as f:
        for text in val_texts:
            f.write(text + "\n")
    
    with open(os.path.join(output_dir, "test_texts.txt"), "w") as f:
        for text in test_texts:
            f.write(text + "\n")
    
    # Save metadata if provided
    if metadata:
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved preprocessed data to {output_dir}")

def load_and_tokenize_data(
    file_path: str,
    tokenizer,
    alpha: float = 10.0,
    precision: int = 3,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    max_length: int = 512,
    stride: int = 256,
    seed: int = 42
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load, preprocess, and tokenize the dataset for training.
    
    Args:
        file_path: Path to the HDF5 file
        tokenizer: Tokenizer to use
        alpha: Scaling parameter
        precision: Decimal precision
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        max_length: Maximum token sequence length
        stride: Stride for sliding window
        seed: Random seed
    
    Returns:
        Tuple of (train_data, val_data, test_data), where each is a tuple of (input_ids, labels)
    """
    # Preprocess the dataset
    train_texts, val_texts, test_texts = preprocess_dataset(
        file_path=file_path,
        alpha=alpha,
        precision=precision,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed
    )
    
    # Tokenize each split
    logger.info("Tokenizing training data...")
    train_input_ids, train_labels = process_sequences_for_training(
        train_texts, tokenizer, max_length, stride
    )
    
    logger.info("Tokenizing validation data...")
    val_input_ids, val_labels = process_sequences_for_training(
        val_texts, tokenizer, max_length, stride
    )
    
    logger.info("Tokenizing test data...")
    test_input_ids, test_labels = process_sequences_for_training(
        test_texts, tokenizer, max_length, stride
    )
    
    return (train_input_ids, train_labels), (val_input_ids, val_labels), (test_input_ids, test_labels)

# For demonstration/testing
if __name__ == "__main__":
    import os
    from pathlib import Path
    import json
    import torch
    from transformers import AutoTokenizer
    
    # Set parameters
    file_path = "data/lotka_volterra_data.h5"
    output_dir = "data/processed"
    alpha = 10.0
    precision = 3
    seed = 42
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Process the data
    logger.info(f"Processing data from {file_path}...")
    train_texts, val_texts, test_texts = preprocess_dataset(
        file_path=file_path,
        alpha=alpha,
        precision=precision,
        seed=seed
    )
    
    # Save the processed data
    logger.info("Saving preprocessed data...")
    metadata = {
        "file_path": file_path,
        "alpha": alpha,
        "precision": precision,
        "seed": seed,
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "test_size": len(test_texts)
    }
    
    save_preprocessed_data(
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts,
        output_dir=output_dir,
        metadata=metadata
    )
    
    # Test tokenization with Qwen tokenizer
    logger.info("Testing tokenization...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Tokenize a sample text
    sample_text = train_texts[0]
    tokens = tokenizer(sample_text, return_tensors="pt").input_ids[0]
    logger.info(f"Sample text tokens: {tokens.shape}")
    
    # Test full tokenization with sliding window
    logger.info("Testing full tokenization with sliding window...")
    (train_ids, train_labels), (val_ids, val_labels), (test_ids, test_labels) = load_and_tokenize_data(
        file_path=file_path,
        tokenizer=tokenizer,
        alpha=alpha,
        precision=precision,
        max_length=512,
        stride=256,
        seed=seed
    )
    
    logger.info(f"Train: {train_ids.shape}, {train_labels.shape}")
    logger.info(f"Val: {val_ids.shape}, {val_labels.shape}")
    logger.info(f"Test: {test_ids.shape}, {test_labels.shape}")
    
    # Print token statistics
    logger.info("Token statistics:")
    logger.info(f"- Percentage of tokens that are padding: {(train_ids == tokenizer.pad_token_id).float().mean().item() * 100:.2f}%")
    logger.info(f"- Number of unique tokens in dataset: {len(torch.unique(train_ids))} / {tokenizer.vocab_size}")
    
    # Verify that the data pipeline works end-to-end
    # Test text back to numeric conversion
    logger.info("Testing text to numeric conversion...")
    numeric_data = text_to_numeric(sample_text)
    logger.info(f"Numeric data shape: {numeric_data.shape}")
    
    logger.info("All tests completed successfully!")