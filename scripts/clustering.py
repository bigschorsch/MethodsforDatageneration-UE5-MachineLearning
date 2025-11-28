from sklearn.metrics import silhouette_score
import pandas as pd
import sklearn
import numpy as np


def prepare_data(df):
    # filtering only numerical features
    df_only_num = df.select_dtypes(include=[np.number])

    # impute NaN values and scale
    imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='median')
    scaler = sklearn.preprocessing.StandardScaler()
    x_imputed = imp.fit_transform(df_only_num)
    x_scaled = scaler.fit_transform(x_imputed)

    return x_scaled


def evaluate_kmeans(X, k_min, k_max):
    sil_scores = []

    for k in range(k_min, k_max):
        kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
        labels = kmeans.labels_
        sil = silhouette_score(X, labels)
        sil_scores.append({"k": k, "silhouette": sil})

    df_scores = pd.DataFrame(sil_scores)
    best = df_scores.loc[df_scores["silhouette"].idxmax()]

    return df_scores, int(best.k), best.silhouette


def evaluate_agglomerative(x_scaled, k_min, k_max):
    sil_scores = []

    # Agglomerative Clustering with different k's
    for k in range(k_min, k_max):
        ag = sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = ag.fit_predict(x_scaled)
        sil = silhouette_score(x_scaled, labels)
        sil_scores.append({"k": k, "silhouette": sil})

    df_scores = pd.DataFrame(sil_scores)
    best = df_scores.loc[df_scores["silhouette"].idxmax()]

    return df_scores, int(best.k), best.silhouette


def evaluate_affinity_propagation(x_scaled):
    sil_scores = []

    # Affinity Propagation with different dampings
    for damping in np.arange(0.5, 1.0, 0.1):
        ap = sklearn.cluster.AffinityPropagation(damping=damping, random_state=0)
        labels = ap.fit_predict(x_scaled)

        # Silhouette braucht >=2 Cluster
        if len(set(labels)) > 1:
            sil = silhouette_score(x_scaled, labels)
        else:
            sil = np.nan

        sil_scores.append({"damping": damping, "silhouette": sil})

    df_scores = pd.DataFrame(sil_scores)

    if df_scores["silhouette"].notna().any():
        best = df_scores.loc[df_scores["silhouette"].idxmax()]
        return df_scores, float(best.damping), best.silhouette

    return df_scores, None, None


def eval_different_clustering(x_scaled):
    kmeans_df, best_k_kmeans, best_sil_kmeans = evaluate_kmeans(x_scaled, 2, 11)
    print("\nKMeans results:")
    print(kmeans_df)
    # best k for kmeans
    print(f"Best k for kmeans: {best_k_kmeans}, silhouette: {best_sil_kmeans:.3f}")

    ag_df, best_k_ag, best_sil_ag = evaluate_agglomerative(x_scaled, 2, 11)
    print("\nAgglomerative results:")
    print(ag_df)
    print(f"Best k for AG: {best_k_ag}, silhouette: {best_sil_ag:.3f}")

    ap_df, best_damping, best_sil_ap = evaluate_affinity_propagation(x_scaled)
    print("\nAffinity Propagation results:")
    print(ap_df)

    if best_damping is not None:
        print(f"Best damping for Affinity Propagation: {best_damping:.1f}, silhouette: {best_sil_ap:.3f}")
    else:
        print("No valid silhouette for Affinity Propagation")
