import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configuration
PHASE1_RESULTS_DIR = "results/lrp_phase1_results"
PHASE2_RESULTS_DIR = "results/phase2_results"
PHASE2_PLOTS_DIR = os.path.join(PHASE2_RESULTS_DIR, "phase2_plots")

def load_R_matrices(npz_path: str):
    """Load all arrays from an .npz file into a dict"""
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}

def prepare_features(R_mat: np.ndarray, use_pca: bool = True, 
                    pca_variance_threshold: float = 0.95) -> np.ndarray:
    """Prepare features with scaling and PCA"""
    scaler = StandardScaler(with_mean=True, with_std=True)
    R_scaled = scaler.fit_transform(R_mat)

    if not use_pca:
        return R_scaled

    pca_full = PCA(n_components=min(R_scaled.shape), svd_solver='full')
    pca_full.fit(R_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum_var, pca_variance_threshold) + 1)

    pca = PCA(n_components=n_comp, svd_solver='auto')
    R_pca = pca.fit_transform(R_scaled)
    return R_pca

def get_K_range(N_neurons: int, k_min: int = 2) -> list:
    """Get range of K values to test"""
    K_max = min(50, max(2, N_neurons // 4))
    return list(range(k_min, K_max + 1))

def evaluate_kmeans(X: np.ndarray, K: int):
    """Evaluate K-Means for a given K"""
    km = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300, verbose=0)
    labels = km.fit_predict(X)
    sse = km.inertia_

    sil = silhouette_score(X, labels) if 1 < K < X.shape[0] else np.nan
    db = davies_bouldin_score(X, labels) if K > 1 else np.nan

    return sse, sil, db

def plot_clustering_curves(results: dict, layer_name: str):
    """Plot clustering curves (Elbow, Silhouette, DBI)"""
    Ks = np.array(results['K'])
    sse_vals = np.array(results['SSE'])
    sil_vals = np.array(results['Silhouette'])
    db_vals = np.array(results['DaviesBouldin'])

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    plt.plot(Ks, sse_vals, '-o', color='tab:blue')
    plt.title(f"{layer_name}: Elbow (SSE vs K)")
    plt.xlabel("K")
    plt.ylabel("SSE (inertia)")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(Ks, sil_vals, '-o', color='tab:green')
    plt.title(f"{layer_name}: Silhouette vs K")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(Ks, db_vals, '-o', color='tab:red')
    plt.title(f"{layer_name}: Davies–Bouldin vs K")
    plt.xlabel("K")
    plt.ylabel("DB Index (lower is better)")
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"Clustering metrics for layer \"{layer_name}\"", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = os.path.join(PHASE2_PLOTS_DIR, f"{layer_name}_clustering_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def choose_optimal_K(results: dict, tolerance: float = 0.03) -> int:
    """Automatically pick optimal K by consensus"""
    Ks = np.array(results['K'])
    sil_vals = np.array(results['Silhouette'])
    db_vals = np.array(results['DaviesBouldin'])
    sse_vals = np.array(results['SSE'])

    # K_sil = argmax silhouette
    valid_sil_mask = ~np.isnan(sil_vals)
    if valid_sil_mask.any():
        arg_sil = np.argmax(sil_vals[valid_sil_mask])
        K_sil = Ks[valid_sil_mask][arg_sil]
    else:
        K_sil = None

    # K_db = argmin DBI
    valid_db_mask = ~np.isnan(db_vals)
    if valid_db_mask.any():
        arg_db = np.argmin(db_vals[valid_db_mask])
        K_db = Ks[valid_db_mask][arg_db]
    else:
        K_db = None

    # Rough elbow
    if len(sse_vals) > 1:
        deltas = sse_vals[:-1] - sse_vals[1:]
        threshold = deltas[0] * 0.1
        elbow_candidates = np.where(deltas < threshold)[0]
        if len(elbow_candidates) > 0:
            idx_e = elbow_candidates[0]
            K_elbow = Ks[idx_e + 1]
        else:
            idx_largest_drop = np.argmax(deltas)
            K_elbow = Ks[idx_largest_drop + 1]
    else:
        K_elbow = Ks[0]

    candidates = [k for k in (K_sil, K_db, K_elbow) if k is not None]
    if len(candidates) == 0:
        return Ks[0]

    median_val = int(np.median(candidates))
    if all(abs(median_val - k) <= 1 for k in candidates):
        return median_val

    if K_sil is not None:
        max_sil = np.nanmax(sil_vals)
        close_idxs = np.where(sil_vals >= (max_sil - tolerance))[0]
        if len(close_idxs) > 0:
            return int(Ks[close_idxs].min())

    return K_elbow

def final_clustering(X: np.ndarray, K: int, method: str = "kmeans"):
    """Run final clustering with chosen K"""
    if method.lower() == "kmeans":
        km = KMeans(n_clusters=K, random_state=42, n_init=20, max_iter=500)
        labels = km.fit_predict(X)
        return labels, km.cluster_centers_
    elif method.lower() == "hierarchical":
        agg = AgglomerativeClustering(n_clusters=K, linkage='ward')
        labels = agg.fit_predict(X)
        return labels, None
    else:
        raise ValueError(f"Unknown method {method!r}")

def run_phase2_clustering(models='both'):
    """Main Phase 2 clustering pipeline"""
    os.makedirs(PHASE2_RESULTS_DIR, exist_ok=True)
    os.makedirs(PHASE2_PLOTS_DIR, exist_ok=True)
    
    fc_npz = os.path.join(PHASE1_RESULTS_DIR, "fc_rmat.npz")
    resnet_npz = os.path.join(PHASE1_RESULTS_DIR, "resnet_rmat.npz")

    print("→ Loading R-matrices from Phase 1 …")
    all_layers = {}
    
    if models in ['both', 'fc'] and os.path.exists(fc_npz):
        fc_data = load_R_matrices(fc_npz)
        for k, v in fc_data.items():
            all_layers[f"fc_{k}"] = v
    
    if models in ['both', 'resnet'] and os.path.exists(resnet_npz):
        resnet_data = load_R_matrices(resnet_npz)
        for k, v in resnet_data.items():
            all_layers[f"res_{k}"] = v

    summary_rows = []
    print("\n→ Beginning Phase 2 clustering …\n")

    for layer_name, R_mat in all_layers.items():
        N_neurons, S_samples = R_mat.shape
        print(f"Layer \"{layer_name}\": {N_neurons} neurons × {S_samples} samples")

        X = prepare_features(R_mat, use_pca=True, pca_variance_threshold=0.95)
        print(f"  • After scaling+PCA → feature‐matrix shape = {X.shape}")

        K_list = get_K_range(N_neurons, k_min=2)
        print(f"  • Searching K in {K_list[:3]} … {K_list[-3:]} (total {len(K_list)} values)")

        results = {'K': [], 'SSE': [], 'Silhouette': [], 'DaviesBouldin': []}
        for K in tqdm(K_list, desc=f"   ▶ Evaluating K-Means for \"{layer_name}\"", leave=False):
            sse, sil, db = evaluate_kmeans(X, K)
            results['K'].append(K)
            results['SSE'].append(sse)
            results['Silhouette'].append(sil)
            results['DaviesBouldin'].append(db)

        plot_clustering_curves(results, layer_name)

        K_opt = choose_optimal_K(results, tolerance=0.03)
        print(f"  → Chosen K* = {K_opt}")

        labels_km, centroids_km = final_clustering(X, K_opt, method="kmeans")
        labels_hc, _ = final_clustering(X, K_opt, method="hierarchical")

        if K_opt > 1:
            sil_km = silhouette_score(X, labels_km)
            db_km = davies_bouldin_score(X, labels_km)
            sil_hc = silhouette_score(X, labels_hc)
            db_hc = davies_bouldin_score(X, labels_hc)
        else:
            sil_km = db_km = sil_hc = db_hc = np.nan

        # Save cluster labels & centroids
        out_labels_km = os.path.join(PHASE2_RESULTS_DIR, f"{layer_name}_kmeans_labels.npy")
        out_labels_hc = os.path.join(PHASE2_RESULTS_DIR, f"{layer_name}_agglo_labels.npy")
        out_centroids = os.path.join(PHASE2_RESULTS_DIR, f"{layer_name}_kmeans_centroids.npy")

        np.save(out_labels_km, labels_km)
        np.save(out_labels_hc, labels_hc)
        np.save(out_centroids, centroids_km)

        summary_rows.append({
            "layer": layer_name,
            "N_neurons": N_neurons,
            "K_opt": K_opt,
            "Silhouette_KMeans": sil_km,
            "DBI_KMeans": db_km,
            "Silhouette_Agglo": sil_hc,
            "DBI_Agglo": db_hc
        })
        
        print(f"    • Saved labels: {layer_name}_kmeans_labels.npy, {layer_name}_agglo_labels.npy")
        print(f"    • Saved centroids: {layer_name}_kmeans_centroids.npy")
        print(f"    • Metrics: Sil_km={sil_km:.4f}, DBI_km={db_km:.4f}, Sil_hc={sil_hc:.4f}, DBI_hc={db_hc:.4f}\n")

    # Write summary CSV
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(PHASE2_RESULTS_DIR, "clustering_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"\nPhase 2 complete! Summary written to: {summary_csv_path}")