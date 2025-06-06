import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict

def analyze_Rmat_statistics(Rmat_dict: Dict[str, np.ndarray], 
                           model_name_prefix: str = "",
                           save_dir: str = None):
    """Analyze and visualize R-matrix statistics"""
    if save_dir is None:
        save_dir = "results/lrp_phase1_results"
    
    print("\n" + "="*30 + f" LRP ANALYSIS for {model_name_prefix} " + "="*30)
    
    for layer_name, R_matrix in Rmat_dict.items():
        if R_matrix.size == 0:
            print(f"\n--- {layer_name.upper()} LAYER: No relevance data.")
            continue
            
        avg_rel_per_feat = np.mean(R_matrix, axis=1)
        print(f"\n--- {layer_name.upper()} LAYER ({R_matrix.shape[0]} Features) ---")
        print(f"  Mean: {np.mean(avg_rel_per_feat):.4f}, Std: {np.std(avg_rel_per_feat):.4f}")
        print(f"  Min/Max: [{np.min(avg_rel_per_feat):.4f}, {np.max(avg_rel_per_feat):.4f}]")
        
        if len(avg_rel_per_feat) > 0:
            q = np.percentile(avg_rel_per_feat, [25, 50, 75])
            print(f"  Quartiles: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}]")
            low_rel_pct = np.sum(avg_rel_per_feat < 0.1) / len(avg_rel_per_feat) * 100
            high_rel_pct = np.sum(avg_rel_per_feat > 0.8) / len(avg_rel_per_feat) * 100
            print(f"  Low Rel (<0.1): {low_rel_pct:.1f}%, High Rel (>0.8): {high_rel_pct:.1f}%")
            
            if len(avg_rel_per_feat) >= 5:
                top_n_idx = np.argsort(avg_rel_per_feat)[-5:][::-1]
                bot_n_idx = np.argsort(avg_rel_per_feat)[:5]
                print(f"  Top 5 Idx: {top_n_idx} ({avg_rel_per_feat[top_n_idx]})")
                print(f"  Bot 5 Idx: {bot_n_idx} ({avg_rel_per_feat[bot_n_idx]})")
        
        # Plot histogram
        plt.figure(figsize=(7, 4))
        plt.hist(avg_rel_per_feat, bins=30, alpha=0.75, edgecolor='black')
        plt.title(f"Avg Feature Relevance - {model_name_prefix} {layer_name}")
        plt.xlabel("Normalized Avg Relevance")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.5)
        plt.savefig(os.path.join(save_dir, f"{model_name_prefix}_{layer_name}_avg_relevance_dist.png"), dpi=150)
        plt.close()

def plot_compression_comparison(original_acc: float, compressed_acc: float,
                              original_params: int, compressed_params: int,
                              save_path: str = None):
    """Plot comparison between original and compressed models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = ['Original', 'Compressed']
    accuracies = [original_acc, compressed_acc]
    colors = ['blue', 'green']
    
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.7)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 105)
    
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Parameter count comparison
    param_counts = [original_params, compressed_params]
    bars2 = ax2.bar(models, param_counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Parameter Count')
    ax2.set_title('Model Size Comparison')
    ax2.ticklabel_format(style='plain', axis='y')
    
    for bar, count in zip(bars2, param_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + original_params*0.01,
                f'{count:,}', ha='center', va='bottom', rotation=0)
    
    # Add compression ratio
    compression_ratio = 100 * compressed_params / original_params
    fig.suptitle(f'Model Compression Results (Compression Ratio: {compression_ratio:.1f}%)', 
                 fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()