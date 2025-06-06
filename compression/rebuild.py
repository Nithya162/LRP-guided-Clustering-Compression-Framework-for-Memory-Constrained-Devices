#!/usr/bin/env python3
"""
Phase-4: Structural compression using K-Reps
Build compressed models using cluster representatives
"""

import os
import pickle
import csv
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Tuple

from models.net import SimpleFCNN, SimpleFCNNCompressed
from data_loaders.mnist_loader import get_mnist_loaders
from utils.metrics import evaluate_model

# Configuration
PHASE3_DIR = "results/phase3_results"
ORIG_MODEL_DIR = "models/saved_models"
OUT_DIR = "results/phase4_results"

PHASE3_PKL = os.path.join(PHASE3_DIR, "SimpleFCNN_kreps.pkl")
PHASE3_CSV = os.path.join(PHASE3_DIR, "SimpleFCNN_kreps_summary.csv")
ORIG_PTH = os.path.join(ORIG_MODEL_DIR, "SimpleFCNN_mnist_phase1.pth")

def load_cluster_memberships():
    """Load cluster memberships from CSV file"""
    memberships = {}
    
    with open(PHASE3_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = row['layer']
            cluster_id = int(row['cluster'])
            num_members = int(row['#members'])
            
            if layer not in memberships:
                memberships[layer] = {}
            memberships[layer][cluster_id] = num_members
    
    return memberships

def build_compressed_state_fixed(rep_table: Dict[str, Dict], 
                                orig_state: dict) -> Tuple[dict, Tuple[int, int, float]]:
    """Build compressed state dict with proper dimension handling"""
    # Load cluster memberships
    try:
        cluster_info = load_cluster_memberships()
    except:
        print("Warning: Could not load cluster memberships")
        cluster_info = {}
    
    new_state = {}
    param_orig = 0
    param_new = 0
    
    # Layer dimensions - original sizes
    layer_dims = {
        'fc1': (28*28, 512),
        'fc2': (512, 256), 
        'fc3': (256, 128),
        'fc4': (128, 64),
        'fc5': (64, 32),
        'fc6': (32, 10)
    }
    
    # Copy fc1 unchanged
    new_state['fc1.weight'] = orig_state['fc1.weight'].clone()
    new_state['fc1.bias'] = orig_state['fc1.bias'].clone()
    param_orig += orig_state['fc1.weight'].numel() + orig_state['fc1.bias'].numel()
    param_new += new_state['fc1.weight'].numel() + new_state['fc1.bias'].numel()
    
    # Process compressible layers fc2-fc5
    compressed_dims = {}
    
    for layer_name in ['fc2', 'fc3', 'fc4', 'fc5']:
        if layer_name not in rep_table:
            continue
            
        reps = rep_table[layer_name]
        K = len(reps)  # Number of clusters = new output dimension
        compressed_dims[layer_name] = K
        
        # Get original dimensions
        orig_in, orig_out = layer_dims[layer_name]
        
        # For input dimension, use compressed size from previous layer
        if layer_name == 'fc2':
            new_in = 512  # fc1 is not compressed
        else:
            prev_layer = f'fc{int(layer_name[-1]) - 1}'
            new_in = compressed_dims[prev_layer]
        
        # Create new weight and bias tensors
        W_new = np.zeros((K, new_in), dtype=np.float32)
        b_new = np.zeros(K, dtype=np.float32)
        
        # Fill with representative weights
        for new_idx, (cluster_id, info) in enumerate(sorted(reps.items())):
            rep_weights = info['weights'].astype(np.float32)
            
            # Handle dimension mismatch
            if layer_name == 'fc2':
                # fc2 takes full 512 input from fc1
                if len(rep_weights) == 512:
                    W_new[new_idx] = rep_weights
                else:
                    # Pad or truncate as needed
                    if len(rep_weights) < 512:
                        W_new[new_idx, :len(rep_weights)] = rep_weights
                    else:
                        W_new[new_idx] = rep_weights[:512]
            else:
                # For other layers, weights should match compressed input dimension
                if len(rep_weights) == new_in:
                    W_new[new_idx] = rep_weights
                elif len(rep_weights) > new_in:
                    # Take first new_in elements
                    W_new[new_idx] = rep_weights[:new_in]
                else:
                    # Pad with zeros
                    W_new[new_idx, :len(rep_weights)] = rep_weights
            
            b_new[new_idx] = float(info['bias'])
        
        # Store tensors
        new_state[f'{layer_name}.weight'] = torch.from_numpy(W_new)
        new_state[f'{layer_name}.bias'] = torch.from_numpy(b_new)
        
        # Update parameter counts
        orig_w = orig_state[f'{layer_name}.weight']
        orig_b = orig_state[f'{layer_name}.bias']
        param_orig += orig_w.numel() + orig_b.numel()
        param_new += W_new.size + b_new.size
        
        print(f"{layer_name}: {orig_in}×{orig_out} -> {new_in}×{K}")
    
    # Handle fc6 (output layer)
    fc6_orig_w = orig_state['fc6.weight'].cpu().numpy()  # (10, 32)
    fc6_orig_b = orig_state['fc6.bias'].cpu().numpy()    # (10,)
    
    K5 = compressed_dims['fc5']
    
    # Create new fc6 weights
    if fc6_orig_w.shape[1] >= K5:
        # Take first K5 columns
        fc6_new_w = fc6_orig_w[:, :K5].astype(np.float32)
    else:
        # Pad if needed
        fc6_new_w = np.zeros((10, K5), dtype=np.float32)
        fc6_new_w[:, :fc6_orig_w.shape[1]] = fc6_orig_w
    
    new_state['fc6.weight'] = torch.from_numpy(fc6_new_w)
    new_state['fc6.bias'] = torch.from_numpy(fc6_orig_b.astype(np.float32))
    
    param_orig += fc6_orig_w.size + fc6_orig_b.size
    param_new += fc6_new_w.size + fc6_orig_b.size
    
    print(f"fc6: 32×10 -> {K5}×10")
    
    return new_state, (param_orig, param_new, 100 * param_new / param_orig)

def run_phase4_compression(models='both'):
    """Run Phase 4: Model compression"""
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if models not in ['both', 'fc']:
        print("Phase 4 currently only supports SimpleFCNN")
        return
    
    print("Phase 4: Structural Compression")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    rep_tbl = pickle.load(open(PHASE3_PKL, "rb"))
    orig_state = torch.load(ORIG_PTH, map_location='cpu')
    
    # Extract new hidden sizes
    k2 = len(rep_tbl['fc2'])
    k3 = len(rep_tbl['fc3']) 
    k4 = len(rep_tbl['fc4'])
    k5 = len(rep_tbl['fc5'])
    print(f"Compressed sizes: fc2={k2}, fc3={k3}, fc4={k4}, fc5={k5}")
    
    # Build new state-dict
    print("Building compressed model...")
    new_state, (p_orig, p_new, p_pct) = build_compressed_state_fixed(rep_tbl, orig_state)
    print(f"Parameters: original={p_orig:,} → compressed={p_new:,} ({p_pct:.1f}% of original)")
    
    # Create compressed model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading compressed model...")
    cm = SimpleFCNNCompressed(k2, k3, k4, k5).to(device)
    
    try:
        cm.load_state_dict(new_state, strict=True)
        print("✓ State dict loaded successfully")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Trying with strict=False...")
        cm.load_state_dict(new_state, strict=False)
    
    cm.eval()
    cm.requires_grad_(False)
    
    # Test accuracy
    print("Testing accuracy...")
    _, test_loader = get_mnist_loaders(batch_size_test=512)
    
    accuracy = evaluate_model(cm, test_loader, "Compressed SimpleFCNN", device)
    
    # Save compressed model
    save_path = os.path.join(OUT_DIR, "SimpleFCNN_compressed.pth")
    torch.save(cm.state_dict(), save_path)
    print(f"✓ Saved compressed model to {save_path}")
    
    # Save compression summary
    summary_path = os.path.join(OUT_DIR, "compression_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("COMPRESSION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Original parameters: {p_orig:,}\n")
        f.write(f"Compressed parameters: {p_new:,}\n")
        f.write(f"Compression ratio: {p_pct:.1f}%\n")
        f.write(f"Space saved: {100-p_pct:.1f}%\n")
        f.write(f"Test accuracy: {accuracy:.2f}%\n")
        f.write("\nLayer-wise compression:\n")
        f.write("fc1: 784×512 (unchanged)\n")
        f.write(f"fc2: 512×256 → 512×{k2}\n")
        f.write(f"fc3: 256×128 → {k2}×{k3}\n")
        f.write(f"fc4: 128×64 → {k3}×{k4}\n")
        f.write(f"fc5: 64×32 → {k4}×{k5}\n")
        f.write(f"fc6: 32×10 → {k5}×10\n")
    
    print(f"\n✓ Summary saved to {summary_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("COMPRESSION SUMMARY")
    print("=" * 50)
    print(f"Original parameters: {p_orig:,}")
    print(f"Compressed parameters: {p_new:,}")
    print(f"Compression ratio: {p_pct:.1f}%")
    print(f"Space saved: {100-p_pct:.1f}%")
    print(f"Test accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    run_phase4_compression()