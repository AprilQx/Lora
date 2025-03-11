"""
Verify that tokenization correctly preserves information.

This script demonstrates that tokenization and decoding are reversible,
ensuring that our preprocessing pipeline doesn't lose information.
"""

import sys
import os
import torch
import h5py
import numpy as np
from transformers import AutoTokenizer

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.processor import load_and_preprocess, numeric_to_text, text_to_numeric

def verify_tokenization(data_path="data/lotka_volterra_data.h5"):
    """
    Verify that tokenization correctly preserves information.
    
    Args:
        data_path: Path to the HDF5 data file
    """
    print("\n===== Tokenization Verification =====\n")
    
    # Step 1: Load the tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Step 2: Load some sample data
    print(f"Loading data from: {data_path}")
    try:
        with h5py.File(data_path, "r") as f:
            # Get just one trajectory
            sample_trajectory = f["trajectories"][0]
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create synthetic data if file doesn't exist
        print("Creating synthetic data instead...")
        sample_trajectory = np.random.rand(100, 2) * 10
    
    # Step 3: Convert to text representation
    print("\nConverting to text representation...")
    alpha = 0.99
    precision = 3
    text_representation= numeric_to_text(sample_trajectory, alpha=alpha, precision=precision)
    
    # Print a small sample
    print(f"Sample text representation (first 50 chars): {text_representation[:50]}...")
    
    # Step 4: Tokenize
    print("\nTokenizing...")
    tokens = tokenizer(text_representation, return_tensors="pt").input_ids[0]
    
    print(f"Token count: {len(tokens)}")
    print(f"Sample tokens (first 20): {tokens[:20].tolist()}")
    
    # Step 5: Decode
    print("\nDecoding back to text...")
    decoded_text = tokenizer.decode(tokens)
    
    print(f"Decoded text length: {len(decoded_text)}")
    print(f"Sample decoded text (first 50 chars): {decoded_text[:50]}...")
    
    # Step 6: Compare original and decoded text
    print("\nComparing original and decoded text...")
    
    # Check if lengths match
    if len(text_representation) == len(decoded_text):
        print(f"✓ Lengths match: {len(text_representation)} characters")
    else:
        print(f"✗ Length mismatch: original={len(text_representation)}, decoded={len(decoded_text)}")
    
    # Check if content matches
    if text_representation == decoded_text:
        print("✓ Content matches exactly!")
    else:
        print("✗ Content differs")
        # Find the first difference
        for i, (orig, dec) in enumerate(zip(text_representation, decoded_text)):
            if orig != dec:
                print(f"  First difference at position {i}: '{orig}' vs '{dec}'")
                break
    
    # Step 7: Convert decoded text back to numeric
    print("\nConverting decoded text back to numeric...")
    numeric_from_decoded = text_to_numeric(decoded_text)
    
    print(f"Reconstructed shape: {numeric_from_decoded.shape}")
    
    
    
    print("\n===== Verification Complete =====")


if __name__ == "__main__":
    # Get data path from command line if provided
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/lotka_volterra_data.h5"
    verify_tokenization(data_path)