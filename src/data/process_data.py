"""
Data processing script for Lotka-Volterra time series data.
Creates train/validation/test splits and saves them as complete sequences.
"""

import os
import json
import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoTokenizer

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from preprocessor import load_data, numeric_to_text, text_to_numeric

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_data_split(
    file_path: str,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
):
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

def save_complete_sequences(
    texts,
    output_file: str,
    subset_name: str
):
    """
    Save complete time series sequences to a text file.
    Each sequence is saved as a single line.
    
    Args:
        texts: List of text sequences
        output_file: Path to output file
        subset_name: Name of the subset (train/val/test) for logging
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text data as complete sequences (one per line)
    with open(output_file, "w") as f:
        for text in texts:
            f.write(text + "\n")
    
    logger.info(f"Saved {len(texts)} complete {subset_name} sequences to {output_file}")

def save_chunked_sequences(
    texts,
    tokenizer,
    output_file: str,
    subset_name: str,
    max_length: int = 512,
    stride: int = 256
):
    """
    Save tokenized chunked sequences to a file in PyTorch format.
    
    Args:
        texts: List of text sequences
        tokenizer: Tokenizer to use
        output_file: Path to output file
        subset_name: Name of the subset (train/val/test) for logging
        max_length: Maximum length of each chunk
        stride: Stride between consecutive chunks
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process sequences into chunks
    input_ids, labels = process_sequences_for_training(texts, tokenizer, max_length, stride)
    
    # Save tokenized data
    torch.save({
        'input_ids': input_ids,
        'labels': labels,
        'metadata': {
            'subset': subset_name,
            'num_original_sequences': len(texts),
            'num_chunks': input_ids.shape[0],
            'max_length': max_length,
            'stride': stride
        }
    }, output_file)
    
    logger.info(f"Saved {input_ids.shape[0]} chunked {subset_name} sequences to {output_file}")
    logger.info(f"  Shape: {input_ids.shape}")

def process_sequences_for_training(
    texts,
    tokenizer,
    max_length: int = 512,
    stride: int = 256
):
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

def process_train_data(
    file_path: str,
    train_indices,
    output_dir: str,
    tokenizer,
    alpha: float = 10.0,
    precision: int = 3,
    max_length: int = 512,
    stride: int = 256
):
    """
    Process training data (with chunking for efficient training).
    
    Args:
        file_path: Path to the HDF5 file
        train_indices: Indices of training trajectories
        output_dir: Directory to save processed data
        tokenizer: Tokenizer to use
        alpha: Scaling parameter
        precision: Decimal precision
        max_length: Maximum sequence length
        stride: Stride for sliding window
    """
    logger.info("Processing training data...")
    
    # Load trajectories
    trajectories, _ = load_data(file_path)
    
    # Convert to text format
    train_texts = [numeric_to_text(trajectories[idx], alpha, precision) for idx in train_indices]
    
    # Save complete sequences (for reference)
    save_complete_sequences(
        texts=train_texts,
        output_file=os.path.join(output_dir, "train_texts.txt"),
        subset_name="training"
    )
    
    # Save chunked sequences (for training)
    save_chunked_sequences(
        texts=train_texts,
        tokenizer=tokenizer,
        output_file=os.path.join(output_dir, "train_tokens.pt"),
        subset_name="training",
        max_length=max_length,
        stride=stride
    )
    
    logger.info("Training data processing complete")

def process_validation_data(
    file_path: str,
    val_indices,
    output_dir: str,
    tokenizer,
    alpha: float = 10.0,
    precision: int = 3,
    max_length: int = 512
):
    """
    Process validation data (for both chunked and complete sequence evaluation).
    
    Args:
        file_path: Path to the HDF5 file
        val_indices: Indices of validation trajectories
        output_dir: Directory to save processed data
        tokenizer: Tokenizer to use
        alpha: Scaling parameter
        precision: Decimal precision
        max_length: Maximum sequence length for chunked evaluation
    """
    logger.info("Processing validation data...")
    
    # Load trajectories
    trajectories, _ = load_data(file_path)
    
    # Convert to text format
    val_texts = [numeric_to_text(trajectories[idx], alpha, precision) for idx in val_indices]
    
    # Save complete sequences (for complete sequence evaluation)
    save_complete_sequences(
        texts=val_texts,
        output_file=os.path.join(output_dir, "val_texts.txt"),
        subset_name="validation"
    )
    
    # Save chunked sequences (for loss evaluation during training)
    save_chunked_sequences(
        texts=val_texts,
        tokenizer=tokenizer,
        output_file=os.path.join(output_dir, "val_tokens.pt"),
        subset_name="validation",
        max_length=max_length,
        stride=max_length  # No overlap for validation
    )
    
    logger.info("Validation data processing complete")

def process_test_data(
    file_path: str,
    test_indices,
    output_dir: str,
    tokenizer,
    alpha: float = 10.0,
    precision: int = 3
):
    """
    Process test data (complete sequences only, no chunking).
    
    Args:
        file_path: Path to the HDF5 file
        test_indices: Indices of test trajectories
        output_dir: Directory to save processed data
        tokenizer: Tokenizer for metadata (not for chunking)
        alpha: Scaling parameter
        precision: Decimal precision
    """
    logger.info("Processing test data...")
    
    # Load trajectories
    trajectories, _ = load_data(file_path)
    
    # Convert to text format and save complete sequences
    test_texts = [numeric_to_text(trajectories[idx], alpha, precision) for idx in test_indices]
    
    # Save complete sequences (for evaluation)
    save_complete_sequences(
        texts=test_texts,
        output_file=os.path.join(output_dir, "test_texts.txt"),
        subset_name="test"
    )
    
    # Save metadata about test data
    test_metadata = {
        "num_sequences": len(test_texts),
        "indices": [int(idx) for idx in test_indices],
        "alpha": alpha,
        "precision": precision,
        "tokenizer": tokenizer.name_or_path
    }
    
    with open(os.path.join(output_dir, "test_metadata.json"), "w") as f:
        json.dump(test_metadata, f, indent=2)
    
    logger.info("Test data processing complete")

def process_all_data(args):
    """
    Process all data splits (train, validation, test).
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Processing data from {args.file_path} with alpha={args.alpha}, precision={args.precision}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data splits
    train_indices, val_indices, test_indices = create_data_split(
        file_path=args.file_path,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Process each split
    if not args.skip_train:
        process_train_data(
            file_path=args.file_path,
            train_indices=train_indices,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            alpha=args.alpha,
            precision=args.precision,
            max_length=args.max_length,
            stride=args.stride
        )
    
    if not args.skip_val:
        process_validation_data(
            file_path=args.file_path,
            val_indices=val_indices,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            alpha=args.alpha,
            precision=args.precision,
            max_length=args.max_length
        )
    
    if not args.skip_test:
        process_test_data(
            file_path=args.file_path,
            test_indices=test_indices,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            alpha=args.alpha,
            precision=args.precision
        )
    
    # Save global metadata
    metadata = {
        "file_path": args.file_path,
        "alpha": args.alpha,
        "precision": args.precision,
        "seed": args.seed,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_indices),
        "tokenizer": args.model_name,
        "max_length": args.max_length,
        "stride": args.stride
    }
    
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"All data processing complete. Results saved in {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process Lotka-Volterra time series data")
    
    # Input/output options
    parser.add_argument("--file_path", type=str, default="data/lotka_volterra_data.h5",
                        help="Path to the HDF5 data file")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory to save processed data")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model name for tokenizer")
    
    # Processing options
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Scaling parameter for numeric values")
    parser.add_argument("--precision", type=int, default=3,
                        help="Decimal precision for text representation")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token sequence length")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride for sliding window in training data")
    
    # Split options
    parser.add_argument("--train_split", type=float, default=0.7,
                        help="Fraction of data for training")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Fraction of data for validation")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Fraction of data for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Skip options
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip processing training data")
    parser.add_argument("--skip_val", action="store_true",
                        help="Skip processing validation data")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip processing test data")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process data
    process_all_data(args)

if __name__ == "__main__":
    main()