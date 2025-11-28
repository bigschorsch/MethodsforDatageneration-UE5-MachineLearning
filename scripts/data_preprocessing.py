import pandas
import matplotlib.pyplot as plt


def get_cleaned_data_frame(path):
    df = pandas.read_csv(path)
    df_isna = df.isna()
    na_ratio = df_isna.mean()
    df_clean = df.loc[:, na_ratio <= 0.30]

    print("number dimensions before cleaning:", df.shape[1])
    print("number of dimensions after cleaning:", df_clean.shape[1])
    print("number of instances (patients):", df.shape[0])
    return df_clean


def analyze_data(df_clean):
    dtypes_df = df_clean.dtypes.to_frame("dtype").reset_index()
    dtypes_df.columns = ["column", "dtype"]

    fig, ax = plt.subplots(figsize=(1, 0.1 * len(dtypes_df)))
    ax.axis("off")
    table = ax.table(
        cellText=dtypes_df.values,
        colLabels=dtypes_df.columns,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(dtypes_df.columns))))
    plt.savefig("pics/dtypes.png", bbox_inches="tight")
