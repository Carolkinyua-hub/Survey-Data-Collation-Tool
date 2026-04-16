import pandas as pd


def cluster_summary(df, labels):
    temp = df.copy()
    temp["cluster"] = labels

    rows = []

    for c in sorted(temp["cluster"].unique()):
        sub = temp[temp["cluster"] == c]

        rows.append({
            "cluster": c,
            "rows": len(sub),
            "pct": round(len(sub) / len(temp) * 100, 1)
        })

    return pd.DataFrame(rows)