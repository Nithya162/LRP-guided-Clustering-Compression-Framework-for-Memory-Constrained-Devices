import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

FC_RMAT_PATH = "results/lrp_phase1_results/fc_rmat.npz"
RESNET_RMAT_PATH = "results/lrp_phase1_results/resnet_rmat.npz"
PHASE2_RESULTS_DIR = "results/phase2_results/phase2_plots_3d"

def plot_3d_clusters(X3: np.ndarray, labels: np.ndarray, relevance_matrix: np.ndarray,
                     method_name: str, layer_name: str, K: int, silhouette: float, 
                     db_index: float, top_n_per_cluster: int = 1):
    """3D scatter plot of neuron clusters"""
    N, _ = relevance_matrix.shape
    avg_relevance = np.mean(relevance_matrix, axis=1)

    if avg_relevance.max() > avg_relevance.min():
        rel_norm = (avg_relevance - avg_relevance.min()) / (avg_relevance.max() - avg_relevance.min())
    else:
        rel_norm = np.zeros_like(avg_relevance)

    sizes = 20 + 80 * rel_norm
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)

    base_cmap = plt.cm.get_cmap("tab20", n_clusters)
    cluster_cmap = ListedColormap(base_cmap.colors[:n_clusters])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=labels, cmap=cluster_cmap,
                    s=sizes, alpha=0.6, edgecolors='white', linewidth=0.5)

    # Highlight top neurons
    all_top_indices = []
    for c in unique_clusters:
        c_mask = (labels == c)
        c_indices = np.where(c_mask)[0]
        if c_indices.size == 0:
            continue
        sub_rels = avg_relevance[c_indices]
        if top_n_per_cluster >= 1:
            idx_sort = np.argsort(sub_rels)[-top_n_per_cluster:]
            top_idxs = c_indices[idx_sort]
            all_top_indices.extend(top_idxs.tolist())

    if all_top_indices:
        ax.scatter(X3[all_top_indices, 0], X3[all_top_indices, 1], X3[all_top_indices, 2],
                  facecolors='none', edgecolors='gold', s=sizes[all_top_indices] + 40,
                  linewidth=1.5, alpha=0.9, label=f"Top {top_n_per_cluster} per cluster")

    # Plot centroids
    for c in unique_clusters:
        c_mask = (labels == c)
        if c_mask.sum() == 0:
            continue
        centroid = X3[c_mask].mean(axis=0)
        centroid_rel = avg_relevance[c_mask].mean()
        ax.scatter(centroid[0], centroid[1], centroid[2], c='red', marker='^',
                  s=200 + 100 * centroid_rel, edgecolors='black', linewidth=1.2, zorder=5)
        ax.text(centroid[0], centroid[1], centroid[2], f"C{c}\n({centroid_rel:.2f})",
               fontsize=9, ha='center', va='center', color='black')

    ax.set_title(f"{layer_name} | {method_name} K={K}  |  Sil={silhouette:.3f}, DBI={db_index:.3f}",
                 fontsize=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.grid(True, alpha=0.3)

    cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, ticks=list(unique_clusters))
    cb.set_label("Cluster ID", rotation=270, labelpad=15)

    plt.tight_layout()
    filename = f"{layer_name}_{method_name}_K{K}_3dscatter.png"
    plt.savefig(os.path.join(PHASE2_RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

def generate_3d_plots_for_rmat_dict(rmat_dict: dict, prefix: str):
    """Generate 3D plots for each layer"""
    os.makedirs(PHASE2_RESULTS_DIR, exist_ok=True)
    
    for layer_name_raw, R_mat in rmat_dict.items():
        layer_name = f"{prefix}_{layer_name_raw.replace('.', '_')}"
        N_neurons, N_samples = R_mat.shape
        print(f"\nProcessing layer '{layer_name}' with shape {N_neurons}×{N_samples}")

        if N_neurons >= 3:
            pca3 = PCA(n_components=3, random_state=42)
            X3 = pca3.fit_transform(R_mat)
        else:
            X3 = np.zeros((N_neurons, 3))

        for Kvis in range(5, 11):
            # K-Means
            km = KMeans(n_clusters=Kvis, random_state=42, n_init=10, max_iter=300)
            km_labels = km.fit_predict(X3)

            if 1 < Kvis < N_neurons:
                sil_km = silhouette_score(X3, km_labels)
                dbi_km = davies_bouldin_score(X3, km_labels)
            else:
                sil_km = np.nan
                dbi_km = np.nan

            plot_3d_clusters(X3, km_labels, R_mat, method_name="KMeans", layer_name=layer_name,
                           K=Kvis, silhouette=sil_km, db_index=dbi_km, top_n_per_cluster=1)

            # Agglomerative
            hc = AgglomerativeClustering(n_clusters=Kvis, linkage="ward")
            hc_labels = hc.fit_predict(X3)

            if 1 < Kvis < N_neurons:
                sil_hc = silhouette_score(X3, hc_labels)
                dbi_hc = davies_bouldin_score(X3, hc_labels)
            else:
                sil_hc = np.nan
                dbi_hc = np.nan

            plot_3d_clusters(X3, hc_labels, R_mat, method_name="Agglo", layer_name=layer_name,
                           K=Kvis, silhouette=sil_hc, db_index=dbi_hc, top_n_per_cluster=1)

            print(f"  • {layer_name} [K={Kvis}]  "
                  f"KMeans → Sil={sil_km:.3f}, DBI={dbi_km:.3f}  |  "
                  f"Agglo → Sil={sil_hc:.3f}, DBI={dbi_hc:.3f}")

if __name__ == "__main__":
    if os.path.exists(FC_RMAT_PATH):
        fc_data = np.load(FC_RMAT_PATH)
        print("Loaded SimpleFCNN R-matrix layers:", fc_data.files)
        generate_3d_plots_for_rmat_dict(fc_data, prefix="fc")

    if os.path.exists(RESNET_RMAT_PATH):
        res_data = np.load(RESNET_RMAT_PATH)
        print("Loaded ResNet9 R-matrix layers:", res_data.files)
        generate_3d_plots_for_rmat_dict(res_data, prefix="res")

    print("\n3D cluster visualizations complete!")