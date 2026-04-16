# app.py
# Lightweight SME Clustering App
# Main features by cluster weighting + requested visualisations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prince
from sklearn.cluster import KMeans

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="SME Clustering App",
    layout="wide"
)

st.title("SME Clustering App")
st.caption(
    "Cluster SMEs and identify the main weighted features driving each cluster."
)

# =====================================================
# SETTINGS
# =====================================================
N_COMPONENTS = 8
N_CLUSTERS = 4
MAX_ROWS = 500

id_cols = [
    "Company name",
    "Company Email",
    "Company number",
    "City"
]

# =====================================================
# MODEL FUNCTION
# =====================================================
@st.cache_data(show_spinner=False)
def run_model(df):

    # Row cap for low memory hosting
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS).copy()

    # IDs
    existing_ids = [c for c in id_cols if c in df.columns]
    ids = df[existing_ids].copy()

    # Remove IDs
    X = df.drop(columns=existing_ids, errors="ignore").copy()

    # Remove constant columns
    X = X.loc[:, X.nunique(dropna=False) > 1]

    # FAMD
    famd = prince.FAMD(
        n_components=N_COMPONENTS,
        random_state=42
    )

    coords = famd.fit_transform(X)

    # KMeans
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(coords)

    # Final results
    result = pd.concat(
        [
            ids.reset_index(drop=True),
            df.reset_index(drop=True),
            pd.Series(labels, name="cluster")
        ],
        axis=1
    )

    return result, coords, labels, X


# =====================================================
# FEATURE IMPORTANCE / WEIGHTING
# =====================================================
def get_cluster_features(result, X):

    temp = X.copy()
    temp["cluster"] = result["cluster"].values

    rows = []

    for cluster_id in sorted(temp["cluster"].unique()):

        cluster_df = temp[temp["cluster"] == cluster_id]
        other_df = temp[temp["cluster"] != cluster_id]

        for col in X.columns:

            # Numeric columns
            if pd.api.types.is_numeric_dtype(X[col]):

                cluster_mean = cluster_df[col].mean()
                overall_mean = other_df[col].mean()

                score = abs(cluster_mean - overall_mean)

                rows.append({
                    "cluster": cluster_id,
                    "feature": col,
                    "score": round(score, 4),
                    "type": "numeric"
                })

            # Categorical columns
            else:

                top_cluster = (
                    cluster_df[col]
                    .astype(str)
                    .value_counts(normalize=True)
                    .head(1)
                )

                if len(top_cluster) > 0:
                    val = top_cluster.index[0]
                    pct = top_cluster.values[0]

                    rows.append({
                        "cluster": cluster_id,
                        "feature": f"{col} = {val}",
                        "score": round(pct, 4),
                        "type": "categorical"
                    })

    feature_df = pd.DataFrame(rows)

    feature_df = feature_df.sort_values(
        ["cluster", "score"],
        ascending=[True, False]
    )

    return feature_df


# =====================================================
# UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =====================================================
# MAIN FLOW
# =====================================================
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        if len(df) > MAX_ROWS:
            st.warning(
                f"Using first {MAX_ROWS} rows for memory efficiency."
            )

        if st.button("Run Analysis"):

            with st.spinner("Running clustering..."):

                result, coords, labels, X = run_model(df)

            # =================================================
            # MAIN FEATURE WEIGHTING
            # =================================================
            st.subheader("Main Features Driving Each Cluster")

            feature_df = get_cluster_features(result, X)

            top_n = st.slider(
                "Top Features Per Cluster",
                3, 10, 5
            )

            display_rows = []

            for c in sorted(feature_df["cluster"].unique()):
                top_features = feature_df[
                    feature_df["cluster"] == c
                ].head(top_n)

                display_rows.append(top_features)

            display_df = pd.concat(display_rows)

            st.dataframe(display_df)

            # =================================================
            # VISUALISATIONS
            # =================================================
            st.subheader("Visualisations")

            viz = st.selectbox(
                "Choose visualisation",
                [
                    "Cluster Counts",
                    "Cluster Map",
                    "City by Cluster",
                    "Cluster Members"
                ]
            )

            # ---------------------------------------------
            # CLUSTER COUNTS
            # ---------------------------------------------
            if viz == "Cluster Counts":

                counts = (
                    result["cluster"]
                    .value_counts()
                    .sort_index()
                )

                st.bar_chart(counts)

            # ---------------------------------------------
            # CLUSTER MAP
            # ---------------------------------------------
            elif viz == "Cluster Map":

                fig, ax = plt.subplots(figsize=(8, 5))

                ax.scatter(
                    coords.iloc[:, 0],
                    coords.iloc[:, 1],
                    c=labels
                )

                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.set_title("Cluster Projection")

                st.pyplot(fig)

            # ---------------------------------------------
            # CITY BY CLUSTER
            # ---------------------------------------------
            elif viz == "City by Cluster":

                if "City" in result.columns:

                    xtab = pd.crosstab(
                        result["City"],
                        result["cluster"]
                    )

                    st.dataframe(xtab)
                    st.bar_chart(xtab)

                else:
                    st.warning("City column not found.")

            # ---------------------------------------------
            # CLUSTER MEMBERS
            # ---------------------------------------------
            elif viz == "Cluster Members":

                selected_cluster = st.selectbox(
                    "Choose Cluster",
                    sorted(result["cluster"].unique())
                )

                st.dataframe(
                    result[
                        result["cluster"] == selected_cluster
                    ]
                )

            # =================================================
            # DOWNLOAD
            # =================================================
            csv = result.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="cluster_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(str(e))

else:
    st.info("Upload CSV file to begin.")
