#!/usr/bin/env python3
"""
Phase-3: K-Rep extraction
Extract cluster representatives using LRP relevance and model weights
"""

import os
import csv
import pickle
import numpy as np
import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Optional

from models.net import SimpleFCNN, ResNet9

# Configuration
PHASE1_DIR = "results/lrp_phase1_results"
PHASE2_DIR = "results/phase2_results"
MODEL_DIR = "models/saved_models"
OUT_DIR = "results/phase3_results"

FC_RMAT = os.path.join(PHASE1_DIR, "fc_rmat.npz")
RES_RMAT = os.path.join(PHASE1_DIR, "resnet_rmat.npz")
FC_MODEL_PTH = os.path.join(MODEL_DIR, "SimpleFCNN_mnist_phase1.pth")
RES_MODEL_PTH = os.path.join(MODEL_DIR, "ResNet9_MNIST_mnist_phase1.pth")

# Weights for scoring formula
W_ALPHA, W_BETA, W_DELTA, W_ETA = 1.0, 1.0, 1.0, 1.0

def load_model(model_cls, pth) -> nn.Module:
    """Load a trained model"""
    m = model_cls()
    m.load_state_dict(torch.load(pth, map_location='cpu'))
    m.eval()
    return m

def flat_W(layer: nn.Module) -> np.ndarray:
    """Flatten weight matrix"""
    W = layer.weight.detach().cpu().numpy()
    return W.reshape(W.shape[0], -1) if isinstance(layer, nn.Conv2d) else W

def get_W_b(model, layer_name) -> Tuple[np.ndarray, np.ndarray]:
    """Extract weights and bias from a layer"""
    layer = dict(model.named_modules())[layer_name]
    W = flat_W(layer)
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.zeros(W.shape[0])
    return W, b

def load_labels(layer: str, model_prefix: str) -> np.ndarray:
    """Load cluster labels for a layer"""
    layer_clean = layer.replace('.', '_')
    layer_with_dots = layer
    
    patterns = [
        f"{model_prefix}{layer_clean}_kmeans_labels.npy",
        f"{model_prefix}{layer_with_dots}_kmeans_labels.npy",
        f"{layer_clean}_kmeans_labels.npy",
        f"{layer_with_dots}_kmeans_labels.npy"
    ]
    
    for pattern in patterns:
        path = os.path.join(PHASE2_DIR, pattern)
        if os.path.exists(path):
            print(f"✓ Found labels: {pattern}")
            return np.load(path)
    
    # Enhanced fallback
    if os.path.exists(PHASE2_DIR):
        for f in os.listdir(PHASE2_DIR):
            if f.endswith('_kmeans_labels.npy'):
                without_suffix = f.replace('_kmeans_labels.npy', '')
                if without_suffix.startswith(model_prefix):
                    file_layer = without_suffix[len(model_prefix):]
                    if (file_layer == layer_clean or 
                        file_layer == layer_with_dots or
                        file_layer.replace('.', '_') == layer_clean or
                        file_layer.replace('_', '.') == layer_with_dots):
                        path = os.path.join(PHASE2_DIR, f)
                        print(f"✓ Found fallback: {f}")
                        return np.load(path)
    
    raise FileNotFoundError(f"Labels file for layer '{layer}' not found in {PHASE2_DIR}")

def load_crit(path) -> Dict[str, np.ndarray]:
    """Load critical path masks (if available)"""
    return dict(np.load(path, allow_pickle=True)) if path and os.path.exists(path) else {}

def rep_for_cluster(idx: np.ndarray, R: np.ndarray, W: np.ndarray, 
                   b: np.ndarray, gamma: np.ndarray) -> dict:
    """
    Select representative for a cluster
    
    Args:
        idx: indices of neurons in this cluster
        R: full relevance matrix [N,S]
        W: flattened weight matrix [N,D]
        b: bias vector [N]
        gamma: critical path indicators [N]
    """
    if idx.size == 0:
        return {}
    
    V_c, R_c, W_c, b_c = R[idx], R[idx].mean(1), W[idx], b[idx]
    c_rel = V_c.mean(0)  # centroid in relevance space
    c_wgt = W_c.mean(0)  # centroid in weight space
    c_bias = b_c.mean()

    # Candidates
    i_rel = np.argmin(np.linalg.norm(V_c - c_rel, axis=1))
    i_wgt = np.argmin(np.linalg.norm(W_c - c_wgt, axis=1))
    i_func = np.argmax(R_c)

    cands = [("rel", i_rel), ("wgt", i_wgt), ("func", i_func), ("synthetic", None)]

    best = None
    bestScore = math.inf
    
    for tag, local in cands:
        if local is None:  # synthetic
            d_rel = d_wgt = 0.0
            rbar = R_c.mean()
            g = gamma[idx].max() if gamma.size else 0.0
            W_rep, b_rep = c_wgt, c_bias
            gidx = None
        else:
            gidx = int(idx[local])
            d_rel = np.linalg.norm(R[gidx] - c_rel)
            d_wgt = np.linalg.norm(W[gidx] - c_wgt)
            rbar = R[gidx].mean()
            g = float(gamma[gidx]) if gamma.size else 0.0
            W_rep, b_rep = W[gidx], b[gidx]

        score = W_ALPHA * d_rel + W_BETA * d_wgt - W_DELTA * rbar - W_ETA * g
        
        if score < bestScore:
            bestScore = score
            best = dict(
                rep_type=tag,
                rep_idx=gidx,
                weights=W_rep.astype(np.float32),
                bias=float(b_rep)
            )
    
    best['members'] = idx.tolist()
    return best

def process_model(model_name: str, rmat_path: str, model_pth: str, crit_mask_path: str = ""):
    """Process a model to extract K-reps"""
    model = load_model(SimpleFCNN if "Simple" in model_name else ResNet9, model_pth)
    R_all = dict(np.load(rmat_path, allow_pickle=True))
    crit = load_crit(crit_mask_path)

    prefix = "fc_" if "Simple" in model_name else "res_"

    out = {}
    for layer, R in R_all.items():
        labels = load_labels(layer.replace(".", "_"), prefix)
        W, b = get_W_b(model, layer)
        gamma = crit.get(layer, np.zeros(R.shape[0], dtype=float))

        reps_layer = {}
        for c in np.unique(labels):
            reps_layer[int(c)] = rep_for_cluster(
                np.where(labels == c)[0], R, W, b, gamma
            )
        
        out[layer] = reps_layer
        print(f"Layer {layer}: stored {len(reps_layer)} reps")

    # Save results
    with open(os.path.join(OUT_DIR, f"{model_name}_kreps.pkl"), "wb") as f:
        pickle.dump(out, f)

    # Save summary CSV
    csv_path = os.path.join(OUT_DIR, f"{model_name}_kreps_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "cluster", "rep_type", "rep_idx", "#members"])

        for layer, reps in out.items():
            labels = load_labels(layer.replace(".", "_"), prefix)
            for cid, info in reps.items():
                w.writerow([
                    layer,
                    cid,
                    info['rep_type'],
                    info['rep_idx'],
                    int((labels == cid).sum())
                ])
    
    print(f"✓ Saved {csv_path}")

def run_phase3_krep_extraction(models='both'):
    """Run Phase 3: K-rep extraction"""
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if models in ['both', 'fc']:
        print("\nProcessing SimpleFCNN...")
        process_model("SimpleFCNN", FC_RMAT, FC_MODEL_PTH)
    
    if models in ['both', 'resnet']:
        print("\nProcessing ResNet9...")
        process_model("ResNet9", RES_RMAT, RES_MODEL_PTH)
    
    print("\nPhase 3 representative extraction finished!")
    print(f"Results saved to {OUT_DIR}")

if __name__ == "__main__":
    run_phase3_krep_extraction()