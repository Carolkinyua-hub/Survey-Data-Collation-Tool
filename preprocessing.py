import pandas as pd

def clean_columns(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("’", "'")
    )
    return df


def preprocess(df):
    df = clean_columns(df).copy()

    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": "Missing", "nan": "Missing"})
        )

    return df