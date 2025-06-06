import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm

FC_RMAT_PATH = "results/lrp_phase1_results/fc_rmat.npz"
RESNET_RMAT_PATH = "results/lrp_phase1_results/resnet_rmat.npz"
OUTPUT_DIR = "results/phase2_results/phase2_combined_plots"
K_VALUES = [6, 7, 8, 9, 10]
TOP_N_PER_CLUSTER = 1
RANDOM_STATE = 42

def compute_pca_embeddings(R_full: np.ndarray, dims: int = 3) -> np.ndarray:
    """Compute PCA embeddings"""
    n_neurons, n_samples = R_full.shape
    if n_neurons < dims:
        padding = np.zeros((n_neurons, dims - n_neurons))
        return np.hstack((R_full, padding))
    pca = PCA(n_components=dims, random_state=RANDOM_STATE)
    return pca.fit_transform(R_full)

def plot_2d_clusters(X2: np.ndarray, labels: np.ndarray, avg_relevance: np.ndarray,
                     layer_prefix: str, model_name: str, K: int, silhouette: float, 
                     db_index: float, top_n: int = TOP_N_PER_CLUSTER):
    """2D scatter plot"""
    N = X2.shape[0]

    if avg_relevance.max() > avg_relevance.min():
        rel_norm = (avg_relevance - avg_relevance.min()) / (avg_relevance.max() - avg_relevance.min())
    else:
        rel_norm = np.zeros_like(avg_relevance)
    sizes = 20 + 80 * rel_norm

    clusters = np.unique(labels)
    n_clusters = len(clusters)
    base_cmap = plt.cm.get_cmap("tab20", n_clusters)
    cluster_cmap = ListedColormap(base_cmap.colors[:n_clusters])

    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap=cluster_cmap, s=sizes,
               alpha=0.6, edgecolors='white', linewidth=0.5)

    # Plot centroids
    for c in clusters:
        mask_c = (labels == c)
        centroid = X2[mask_c].mean(axis=0)
        centroid_rel = avg_relevance[mask_c].mean() if mask_c.sum() > 0 else 0.0
        plt.scatter(centroid[0], centroid[1], c='red', marker='^',
                   s=200 + 100 * centroid_rel, edgecolors='black', linewidth=1.2, zorder=5)
        plt.text(centroid[0], centroid[1], f"C{c}\n({centroid_rel:.2f})",
                fontsize=8, ha='center', va='center', color='black',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Highlight top neurons
    top_indices = []
    for c in clusters:
        mask_c = (labels == c)
        idxs_c = np.where(mask_c)[0]
        if idxs_c.size == 0:
            continue
        rels_c = avg_relevance[idxs_c]
        top_n_c = min(top_n, idxs_c.size)
        top_sort_idx = np.argsort(rels_c)[-top_n_c:]
        top_idx = idxs_c[top_sort_idx]
        top_indices.extend(top_idx.tolist())

    if top_indices:
        plt.scatter(X2[top_indices, 0], X2[top_indices, 1], facecolors='none',
                   edgecolors='gold', s=sizes[top_indices] + 40, linewidth=1.5,
                   alpha=0.9, label=f"Top {top_n} per cluster")

    plt.title(f"{model_name} (combined) | 2D PCA | K={K}  |  Sil={silhouette:.3f}, DBI={db_index:.3f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    fname = f"{model_name}_combined_K{K}_2D.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()

def plot_3d_clusters(X3: np.ndarray, labels: np.ndarray, avg_relevance: np.ndarray,
                     layer_prefix: str, model_name: str, K: int, silhouette: float, 
                     db_index: float, top_n: int = TOP_N_PER_CLUSTER):
    """3D scatter plot"""
    N = X3.shape[0]

    if avg_relevance.max() > avg_relevance.min():
        rel_norm = (avg_relevance - avg_relevance.min()) / (avg_relevance.max() - avg_relevance.min())
    else:
        rel_norm = np.zeros_like(avg_relevance)
    sizes = 20 + 80 * rel_norm

    clusters = np.unique(labels)
    n_clusters = len(clusters)
    base_cmap = plt.cm.get_cmap("tab20", n_clusters)
    cluster_cmap = ListedColormap(base_cmap.colors[:n_clusters])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=labels, cmap=cluster_cmap,
                    s=sizes, alpha=0.6, edgecolors='white', linewidth=0.5)

    # Plot centroids
    for c in clusters:
        mask_c = (labels == c)
        if mask_c.sum() == 0:
            continue
        centroid = X3[mask_c].mean(axis=0)
        centroid_rel = avg_relevance[mask_c].mean()
        ax.scatter(centroid[0], centroid[1], centroid[2], c='red', marker='^',
                  s=200 + 100 * centroid_rel, edgecolors='black', linewidth=1.2, zorder=5)
        ax.text(centroid[0], centroid[1], centroid[2], f"C{c}\n({centroid_rel:.2f})",
               fontsize=8, ha='center', va='center', color='black',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Highlight top neurons
    top_indices = []
    for c in clusters:
        mask_c = (labels == c)
        idxs_c = np.where(mask_c)[0]
        if idxs_c.size == 0:
            continue
        rels_c = avg_relevance[idxs_c]
        top_n_c = min(top_n, idxs_c.size)
        top_sort_idx = np.argsort(rels_c)[-top_n_c:]
        top_idx = idxs_c[top_sort_idx]
        top_indices.extend(top_idx.tolist())

    if top_indices:
        ax.scatter(X3[top_indices, 0], X3[top_indices, 1], X3[top_indices, 2],
                  facecolors='none', edgecolors='gold', s=sizes[top_indices] + 40,
                  linewidth=1.5, alpha=0.9, label=f"Top {top_n} per cluster")

    ax.set_title(f"{model_name} (combined) | 3D PCA | K={K}  |  Sil={silhouette:.3f}, DBI={db_index:.3f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    fname = f"{model_name}_combined_K{K}_3D.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()

def run_combined_clustering(rmat_path: str, model_name: str):
    """Run combined clustering on all layers"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    npz_data = np.load(rmat_path)
    layer_names = npz_data.files
    print(f"Loaded {model_name} R-matrix layers: {layer_names}")

    # Concatenate all layers
    mats = []
    for layer in layer_names:
        mat = npz_data[layer]
        if mat.ndim == 2 and mat.size > 0:
            mats.append(mat)
    
    if len(mats) == 0:
        print(f"Warning: No valid R-matrices found for {model_name}. Skipping.")
        return

    # Ensure all have same number of columns
    n_cols = mats[0].shape[1]
    for i in range(len(mats)):
        if mats[i].shape[1] != n_cols:
            cols_i = mats[i].shape[1]
            if cols_i > n_cols:
                mats[i] = mats[i][:, :n_cols]
            else:
                padding = np.zeros((mats[i].shape[0], n_cols - cols_i))
                mats[i] = np.hstack((mats[i], padding))

    R_full = np.vstack(mats)
    N_neurons, N_samples = R_full.shape
    print(f"Combined R_full shape for {model_name}: {N_neurons} neurons × {N_samples} samples")

    avg_relevance = np.mean(R_full, axis=1)

    # PCA embeddings
    X2 = compute_pca_embeddings(R_full, dims=2)
    X3 = compute_pca_embeddings(R_full, dims=3)

    # Cluster for each K
    for K in K_VALUES:
        if K < 2 or K > N_neurons:
            continue

        print(f"\nClustering {model_name} combined with KMeans, K={K} ...")
        kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(R_full)

        if K > 1:
            sil = silhouette_score(R_full, labels)
            dbi = davies_bouldin_score(R_full, labels)
        else:
            sil = np.nan
            dbi = np.nan

        plot_2d_clusters(X2, labels, avg_relevance, layer_prefix="combined", 
                        model_name=model_name, K=K, silhouette=sil, 
                        db_index=dbi, top_n=TOP_N_PER_CLUSTER)

        plot_3d_clusters(X3, labels, avg_relevance, layer_prefix="combined", 
                        model_name=model_name, K=K, silhouette=sil, 
                        db_index=dbi, top_n=TOP_N_PER_CLUSTER)

        print(f"  Done K={K}: Silhouette={sil:.3f}, Davies–Bouldin={dbi:.3f}")

if __name__ == "__main__":
    if os.path.exists(FC_RMAT_PATH):
        run_combined_clustering(FC_RMAT_PATH, model_name="SimpleFCNN")
    else:
        print(f"Warning: {FC_RMAT_PATH} not found.")

    if os.path.exists(RESNET_RMAT_PATH):
        run_combined_clustering(RESNET_RMAT_PATH, model_name="ResNet9")
    else:
        print(f"Warning: {RESNET_RMAT_PATH} not found.")

    print("\nCombined clustering and 2D/3D plots complete!")