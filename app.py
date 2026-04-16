# app.py
# SME Clustering App
# Upload CSV -> remove IDs from backend -> cluster -> reattach IDs to results

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import prince
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="SME Clustering App",
    layout="wide"
)

st.title("SME Clustering App")
st.caption(
    "Upload survey or SME data, remove identifier columns from modeling, "
    "cluster records, then reattach IDs to results."
)

# =====================================================
# HELPERS
# =====================================================

def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("’", "'")
    )
    return df


def preprocess(df):
    """
    Clean object columns and missing values.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"": "Missing", "nan": "Missing", "None": "Missing"})
            )
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def auto_select_k(X, min_k=2, max_k=8):
    """
    Choose best K using silhouette score.
    """
    best_k = 2
    best_score = -1

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        # avoid invalid silhouette edge case
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_k = k

    return best_k, best_score


def top_profile(df, labels):
    """
    Basic cluster profile summary.
    """
    temp = df.copy()
    temp["cluster"] = labels

    rows = []

    for c in sorted(temp["cluster"].unique()):
        sub = temp[temp["cluster"] == c]

        rows.append({
            "cluster": c,
            "records": len(sub),
            "share_%": round(len(sub) / len(temp) * 100, 1)
        })

    return pd.DataFrame(rows)


# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    df = pd.read_csv(uploaded_file)
    df = clean_columns(df)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # -------------------------------------------------
    # SELECT ID COLUMNS
    # -------------------------------------------------
    st.subheader("Step 1: Select Identifier Columns")

    st.caption(
        "Choose columns like Company Name, Email, Phone, Registration No. "
        "These will be excluded from clustering and reattached later."
    )

    id_cols = st.multiselect(
        "Identifier Columns",
        options=df.columns.tolist()
    )

    # -------------------------------------------------
    # MODEL SETTINGS
    # -------------------------------------------------
    st.subheader("Step 2: Analysis Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_components = st.slider(
            "FAMD Components",
            min_value=2,
            max_value=15,
            value=5
        )

    with col2:
        mode = st.radio(
            "Cluster Selection",
            ["Auto", "Manual"]
        )

    with col3:
        if mode == "Manual":
            manual_k = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=10,
                value=4
            )
        else:
            manual_k = None

    # -------------------------------------------------
    # RUN BUTTON
    # -------------------------------------------------
    if st.button("Run Clustering"):

        with st.spinner("Processing data and training model..."):

            # -----------------------------------------
            # STORE IDS
            # -----------------------------------------
            ids = df[id_cols].copy() if len(id_cols) > 0 else pd.DataFrame(index=df.index)

            # -----------------------------------------
            # REMOVE IDS FROM BACKEND
            # -----------------------------------------
            X = df.drop(columns=id_cols, errors="ignore").copy()

            # -----------------------------------------
            # PREPROCESS
            # -----------------------------------------
            X = preprocess(X)

            # -----------------------------------------
            # FAMD
            # -----------------------------------------
            famd = prince.FAMD(
                n_components=n_components,
                random_state=42
            )

            X_famd = famd.fit_transform(X)

            # -----------------------------------------
            # CHOOSE K
            # -----------------------------------------
            if mode == "Auto":
                k, score = auto_select_k(X_famd)

                st.success(
                    f"Auto-selected clusters = {k} | "
                    f"Silhouette Score = {score:.3f}"
                )
            else:
                k = manual_k

            # -----------------------------------------
            # CLUSTER
            # -----------------------------------------
            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            labels = kmeans.fit_predict(X_famd)

        # =================================================
        # RESULTS
        # =================================================

        st.subheader("Cluster Summary")

        summary = top_profile(df, labels)
        st.dataframe(summary)

        # ---------------------------------------------
        # REATTACH IDS TO RESULTS
        # ---------------------------------------------
        result = pd.concat(
            [
                ids.reset_index(drop=True),
                df.drop(columns=id_cols, errors="ignore").reset_index(drop=True),
                pd.Series(labels, name="cluster")
            ],
            axis=1
        )

        st.subheader("Clustered Results")
        st.dataframe(result.head(50))

        # ---------------------------------------------
        # COUNTS
        # ---------------------------------------------
        st.subheader("Cluster Counts")
        counts = result["cluster"].value_counts().sort_index()
        st.bar_chart(counts)

        # ---------------------------------------------
        # SCATTER PLOT
        # ---------------------------------------------
        st.subheader("Cluster Map")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(
            X_famd.iloc[:, 0],
            X_famd.iloc[:, 1],
            c=labels
        )

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title("FAMD Cluster Projection")

        st.pyplot(fig)

        # ---------------------------------------------
        # FILTER BY CLUSTER
        # ---------------------------------------------
        st.subheader("View Specific Cluster")

        selected_cluster = st.selectbox(
            "Choose Cluster",
            sorted(result["cluster"].unique())
        )

        filtered = result[result["cluster"] == selected_cluster]

        st.dataframe(filtered)

        # ---------------------------------------------
        # DOWNLOAD
        # ---------------------------------------------
        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Clustered CSV",
            data=csv,
            file_name="clustered_results.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a CSV file to begin.")
