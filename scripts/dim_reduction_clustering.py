import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def run_pca_and_cluster(x_scaled, best_k):
    # PCA (2 Components)
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(X=x_scaled)

    plt.figure(figsize=(6, 5))
    plt.scatter(x_pca[:, 0], x_pca[:, 1])
    plt.title("PCA (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("pics/pca.png")
    plt.close()

    # clustering
    kmeans_pca = KMeans(n_clusters=best_k, random_state=0, n_init="auto")
    labels_pca = kmeans_pca.fit_predict(x_pca)

    plt.figure(figsize=(6, 5))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels_pca, cmap="viridis")
    plt.title(f"PCA + KMeans (k={best_k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("pics/pca_clustered.png")
    plt.close()
    return x_pca, labels_pca


def run_tsne_and_cluster(x_scaled, best_k):
    # t-SNE (2 Components)
    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    x_tsne = tsne.fit_transform(x_scaled)

    plt.figure(figsize=(6, 5))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    plt.title("t-SNE (2D)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig("pics/tsne.png")
    plt.close()

    # clustering
    kmeans_tsne = KMeans(n_clusters=best_k, random_state=0, n_init="auto")
    labels_tsne = kmeans_tsne.fit_predict(x_tsne)

    plt.figure(figsize=(6, 5))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels_tsne, cmap="viridis")
    plt.title(f"t-SNE + KMeans (k={best_k})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig("pics/tsne_clustered.png")
    plt.close()
    return x_tsne, labels_tsne


def encode_glucose_in_scatterplot(reduced_x, cluster_labels, df, method_name, save_path):
    glucose_col = '2345-7_Glucose [Mass/volume] in Serum or Plasma'

    if glucose_col not in df.columns:
        raise ValueError(f"Column '{glucose_col}' not found in dataframe.")

    glucose_values = df[glucose_col]

    glucose_values = glucose_values.fillna(glucose_values.median())

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        reduced_x[:, 0],
        reduced_x[:, 1],
        c=glucose_values,
        cmap="viridis",
        alpha=0.85,
        s=40,
        edgecolor="none"
    )

    plt.title(f"{method_name} + Clusters + Glucose Encoding")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    for cluster_id in np.unique(cluster_labels):
        cluster_points = reduced_x[cluster_labels == cluster_id]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            edgecolor="black",
            facecolor="none",
            s=70,
            linewidths=0.3
        )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Glucose level")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
