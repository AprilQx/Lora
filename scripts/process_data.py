"""
Process Lotka-Volterra data, tokenize it, and save to files for later use.
"""

import os
import torch
import argparse
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessor import load_and_preprocess


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
    train_tokens, val_tokens = load_and_preprocess(
        data_path,
        tokenizer=tokenizer,
        val_split=val_split,
        alpha=alpha,
        precision=precision,
        seed=seed,
        max_length=max_length,
        stride=stride
    )
    
    # Save tokenized data
    torch.save(train_tokens, os.path.join(output_dir, "train_tokens.pt"))
    torch.save(val_tokens, os.path.join(output_dir, "val_tokens.pt"))
    
    print(f"Tokenized data saved to {output_dir}")
    print(f"Train tokens shape: {train_tokens.shape}")
    print(f"Val tokens shape: {val_tokens.shape}")
    
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
        "train_tokens_shape": list(train_tokens.shape),
        "val_tokens_shape": list(val_tokens.shape)
    }
    
    # Save configuration as well
    import json
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