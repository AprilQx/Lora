"""
Process Lotka-Volterra data, tokenize it, and save to files for later use.
"""

import os
import torch
import argparse
import json
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocessor import load_data, scale_data, numeric_to_text


def load_and_preprocess(
    file_path: str,
    tokenizer=None,
    val_split: float = 0.2,
    alpha: float = 10,
    precision: int = 3,
    seed: int = 42,
    max_length: int = 512,
    stride: int = 256,
):
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
        If tokenizer is None: Tuple of (train_texts, val_texts) - lists of text sequences
        If tokenizer is provided: Tuple of (train_tokens, val_tokens) - processed tokens
    """
    # Load the data
    trajectories, time_points = load_data(file_path)
    
    # Split the data into training and validation sets
    import numpy as np
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
        text = numeric_to_text(trajectory, alpha=alpha, precision=precision)
        train_texts.append(text)
    
    val_texts = []
    for trajectory in val_trajectories:
        text = numeric_to_text(trajectory, alpha=alpha, precision=precision)
        val_texts.append(text)
    
    if tokenizer is not None:
        train_tokens = process_sequences(train_texts, tokenizer, max_length, stride)
        val_tokens = process_sequences(val_texts, tokenizer, max_length, stride)
        return train_tokens, val_tokens
    
    return train_texts, val_texts


def process_sequences(texts, tokenizer, max_length=512, stride=256):
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


def process_and_save_data(
    data_path, 
    output_dir="data/processed", 
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    alpha=10, 
    precision=3, 
    val_split=0.2,
    max_length=512, 
    stride=256, 
    seed=42
):
    """
    Process the data, tokenize it, and save to files.
    
    Args:
        data_path: Path to the HDF5 data file
        output_dir: Directory to save the processed data
        model_name: Name of the model to use for tokenization
        alpha: Alpha parameter for data scaling
        precision: Decimal precision for numeric representation
        val_split: Validation split ratio
        max_length: Maximum sequence length
        stride: Stride for chunking sequences
        seed: Random seed
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"Processing data from: {data_path}")
    
    # First, process and save the text representations
    print("Converting to text format...")
    train_texts, val_texts = load_and_preprocess(
        data_path,
        val_split=val_split,
        alpha=alpha,
        precision=precision,
        seed=seed
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text representations
    with open(os.path.join(output_dir, "train_texts.txt"), "w") as f:
        for text in train_texts:
            f.write(text + "\n")
            
    with open(os.path.join(output_dir, "val_texts.txt"), "w") as f:
        for text in val_texts:
            f.write(text + "\n")
            
    print(f"Text representations saved to {output_dir}")
    
    # Now tokenize and save the tokenized data
    print("Tokenizing data...")
    token_data = load_and_preprocess(
        data_path,
        tokenizer=tokenizer,
        val_split=val_split,
        alpha=alpha,
        precision=precision,
        seed=seed,
        max_length=max_length,
        stride=stride
    )
    
    # Unpack token data
    train_tokens, val_tokens = token_data
    
    # Save tokenized data
    torch.save(train_tokens, os.path.join(output_dir, "train_tokens.pt"))
    torch.save(val_tokens, os.path.join(output_dir, "val_tokens.pt"))
    
    print(f"Tokenized data saved to {output_dir}")
    print(f"Train tokens count: {len(train_tokens)}")
    print(f"Val tokens count: {len(val_tokens)}")
    
    # Save metadata
    metadata = {
        "alpha": alpha,
        "precision": precision,
        "val_split": val_split,
        "max_length": max_length,
        "stride": stride,
        "seed": seed,
        "model_name": model_name,
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "train_chunks": len(train_tokens),
        "val_chunks": len(val_tokens)
    }
    
    # Save configuration as well
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {output_dir}/metadata.json")
    print("Processing complete!")
    
    return {
        "train_texts": train_texts,
        "val_texts": val_texts,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "metadata": metadata
    }


def main():
    parser = argparse.ArgumentParser(description="Process and save tokenized data")
    parser.add_argument("--data_path", type=str, default="data/lotka_volterra_data.h5", 
                        help="Path to data file")
    parser.add_argument("--output_dir", type=str, default="data/processed", 
                        help="Directory to save processed data")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", 
                        help="Model name for tokenizer")
    parser.add_argument("--alpha", type=float, default=10, 
                        help="Alpha parameter for data scaling")
    parser.add_argument("--precision", type=int, default=3, 
                        help="Decimal precision for numeric representation")
    parser.add_argument("--val_split", type=float, default=0.2, 
                        help="Validation split ratio")
    parser.add_argument("--max_length", type=int, default=512, 
                        help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=256, 
                        help="Stride for chunking sequences")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    
    args = parser.parse_args()
    
    process_and_save_data(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        alpha=args.alpha,
        precision=args.precision,
        val_split=args.val_split,
        max_length=args.max_length,
        stride=args.stride,
        seed=args.seed
    )
    

if __name__ == "__main__":
    main()