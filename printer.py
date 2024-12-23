import polars as pl

def format_classification_report(file_path):
    df = pl.read_csv(file_path)

    df = df.select(
        [
            pl.col("precision"),
            pl.col("recall"),
            pl.col("f1-score"),
            pl.col("support")
        ]
    )

    for row in df.iter_rows(named=True):
        print(
            f"Precision: {row['precision']:.2f}, "
            f"Recall: {row['recall']:.2f}, "
            f"F1-Score: {row['f1-score']:.2f}, "
            f"Support: {int(row['support'])}"
        )

file_path = "./classification_report.csv"
format_classification_report(file_path)
