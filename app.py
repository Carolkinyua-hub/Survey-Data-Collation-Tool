import streamlit as st
import pandas as pd
import prince
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Survey Clustering Studio")

# -------------------
# Helpers
# -------------------

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

    # object cleanup
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": "Missing", "": "Missing"})
        )

    return df

def auto_k(X, k_min=2, k_max=8):
    best_k = 2
    best_score = -1

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score

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

# -------------------
# UI
# -------------------

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)
    df = preprocess(df)

    st.subheader("Preview")
    st.dataframe(df.head())

    id_cols = st.multiselect(
        "Select ID columns (optional)",
        options=df.columns.tolist()
    )

    X = df.drop(columns=id_cols, errors="ignore")

    famd_components = st.slider(
        "FAMD Components",
        min_value=2,
        max_value=15,
        value=8
    )

    cluster_mode = st.radio(
        "Cluster Selection",
        ["Auto-select", "Manual"]
    )

    if cluster_mode == "Manual":
        k = st.slider("Choose number of clusters", 2, 10, 4)
    else:
        k = None

    save_model = st.checkbox("Save trained model")

    if st.button("Run Clustering"):

        with st.spinner("Training FAMD..."):

            famd = prince.FAMD(
                n_components=famd_components,
                random_state=42
            )

            X_famd = famd.fit_transform(X)

        if cluster_mode == "Auto-select":
            k, score = auto_k(X_famd)
            st.success(f"Auto-selected k={k} | silhouette={score:.3f}")

        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = km.fit_predict(X_famd)

        # results
        result = df.copy()
        result["cluster"] = labels

        st.subheader("Cluster Summary")
        st.dataframe(cluster_summary(df, labels))

        st.subheader("Cluster Counts")
        st.bar_chart(result["cluster"].value_counts().sort_index())

        # scatter
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(X_famd.iloc[:,0], X_famd.iloc[:,1], c=labels)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title("FAMD Cluster Map")
        st.pyplot(fig)

        # top categories by chosen field
        cat_cols = X.select_dtypes(include="object").columns.tolist()

        if cat_cols:
            field = st.selectbox("Compare clusters by field", cat_cols)

            cross = pd.crosstab(
                result["cluster"],
                result[field],
                normalize="index"
            )

            st.subheader(f"Cluster vs {field}")
            st.dataframe(cross)

        # download
        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results CSV",
            csv,
            "clustered_results.csv",
            "text/csv"
        )

        # optional save
        if save_model:
            joblib.dump(famd, "famd.joblib")
            joblib.dump(km, "kmeans.joblib")
            st.success("Models saved locally.")
