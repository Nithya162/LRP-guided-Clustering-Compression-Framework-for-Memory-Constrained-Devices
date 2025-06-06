#!/usr/bin/env python3
"""
LRP Guided Clustering for Neural Network Compression - Main Entry Point
Runs all phases of the LRP-based neural network compression pipeline
"""

import os
import sys
import argparse
import torch
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Import phase functions
from lrp.lrp import run_phase1_lrp
from clustering.clustering import run_phase2_clustering
from compression.krep import run_phase3_krep_extraction
from compression.rebuild import run_phase4_compression

def main():
    parser = argparse.ArgumentParser(description='LAPAC Quantization Pipeline')
    parser.add_argument('--phase', type=str, choices=['all', '1', '2', '3', '4'], 
                       default='all', help='Which phase(s) to run')
    parser.add_argument('--model', type=str, choices=['fc', 'resnet', 'both'], 
                       default='both', help='Which model(s) to process')
    parser.add_argument('--force-train', action='store_true', 
                       help='Force retraining even if saved models exist')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print("="*80)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Run phases
    if args.phase in ['all', '1']:
        print("\n=== PHASE 1: LRP Analysis ===")
        run_phase1_lrp(models=args.model, force_train=args.force_train, device=device)
    
    if args.phase in ['all', '2']:
        print("\n=== PHASE 2: Clustering ===")
        run_phase2_clustering(models=args.model)
    
    if args.phase in ['all', '3']:
        print("\n=== PHASE 3: K-Rep Extraction ===")
        run_phase3_krep_extraction(models=args.model)
    
    if args.phase in ['all', '4']:
        print("\n=== PHASE 4: Model Compression ===")
        run_phase4_compression(models=args.model)
    
    print("\n" + "="*80)
    print("Pipeline execution completed!")
    
    # Print summary
    if args.phase == 'all':
        print("\nResults Summary:")
        print("- Phase 1 outputs: results/lrp_phase1_results/")
        print("- Phase 2 outputs: results/phase2_results/")
        print("- Phase 3 outputs: results/phase3_results/")
        print("- Phase 4 outputs: results/phase4_results/")
        print("- Saved models: models/saved_models/")

if __name__ == "__main__":
    main()