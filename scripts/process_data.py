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
from src.data.preprocessor import load_data, scale_data, numeric_to_text, load_and_preprocess



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