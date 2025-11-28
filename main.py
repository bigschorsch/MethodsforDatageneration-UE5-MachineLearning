from scripts.data_preprocessing import *
from scripts.clustering import *
from scripts.dim_reduction_clustering import *

## Exercise a)
cleaned_df = get_cleaned_data_frame("data/data_ehr.csv")

## Exercise b) --> solution saved in /pics/dtypes.png
analyze_data(cleaned_df)

## Exercise c)
x_scaled = prepare_data(cleaned_df)
eval_different_clustering(x_scaled)

# best clustering method is kmeans with k = 2
best_k = 2

## Exercise d) and e) --> solution saved in /pics/pca... and /pics/tsne...
x_pca, labels_pca = run_pca_and_cluster(x_scaled, best_k)
x_tsne, labels_tsne = run_tsne_and_cluster(x_scaled, best_k)

## Exercise f)
encode_glucose_in_scatterplot(
    reduced_x=x_pca,
    cluster_labels=labels_pca,
    df=cleaned_df,
    method_name="PCA",
    save_path="pics/pca_glucose.png"
)
encode_glucose_in_scatterplot(
    reduced_x=x_tsne,
    cluster_labels=labels_tsne,
    df=cleaned_df,
    method_name="t-SNE",
    save_path="pics/tsne_glucose.png"
)
