# app.py
# Lightweight SME Clustering App
# Includes:
# 1. Main weighted cluster drivers
# 2. Descriptive categorical charts
# 3. Cluster visualisations
# 4. Final company ID + cluster table

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import prince
from sklearn.cluster import KMeans

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="SME Clustering App", layout="wide")

st.title("SME Clustering App")
st.caption(
    "Cluster SMEs, understand cluster drivers, and review descriptive insights."
)

# =====================================================
# SETTINGS
# =====================================================
N_COMPONENTS = 2
N_CLUSTERS = 4
MAX_ROWS = 500

id_cols = [
    "Company name",
    "Company Email",
    "Company number",
    "City"
]

selected_cat_features = [
    "1. In general, how do you assess your company's performance in 2023 compared to 2022?",
    "102. Type of company",
    "Recoded CA 2023",
    "Gender",
    "Status",
]

# =====================================================
# MODEL FUNCTION
# =====================================================
@st.cache_data(show_spinner=False)
def run_model(df):

    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS).copy()

    existing_ids = [c for c in id_cols if c in df.columns]
    ids = df[existing_ids].copy()

    X = df.drop(columns=existing_ids, errors="ignore").copy()

    # Remove constant columns
    X = X.loc[:, X.nunique(dropna=False) > 1]

    famd = prince.FAMD(
        n_components=N_COMPONENTS,
        random_state=42
    )

    coords = famd.fit_transform(X)

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(coords)

    result = pd.concat(
        [
            ids.reset_index(drop=True),
            pd.Series(labels, name="cluster")
        ],
        axis=1
    )

    return result, coords, labels, X


# =====================================================
# FEATURE WEIGHTING
# =====================================================
def get_cluster_features(result, X):

    temp = X.copy()
    temp["cluster"] = result["cluster"].values

    rows = []

    for cluster_id in sorted(temp["cluster"].unique()):

        cluster_df = temp[temp["cluster"] == cluster_id]
        other_df = temp[temp["cluster"] != cluster_id]

        for col in X.columns:

            if pd.api.types.is_numeric_dtype(X[col]):

                score = abs(
                    cluster_df[col].mean() -
                    other_df[col].mean()
                )

                rows.append({
                    "cluster": cluster_id,
                    "feature": col,
                    "score": round(score, 4)
                })

            else:

                top = (
                    cluster_df[col]
                    .astype(str)
                    .value_counts(normalize=True)
                    .head(1)
                )

                if len(top) > 0:
                    rows.append({
                        "cluster": cluster_id,
                        "feature": f"{col} = {top.index[0]}",
                        "score": round(top.values[0], 4)
                    })

    out = pd.DataFrame(rows)

    return out.sort_values(
        ["cluster", "score"],
        ascending=[True, False]
    )


# =====================================================
# FILE UPLOAD
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
                f"Using first {MAX_ROWS} rows for performance."
            )

        if st.button("Run Analysis"):

            with st.spinner("Running clustering..."):

                result, coords, labels, X = run_model(df)

            # =================================================
            # MAIN FEATURES
            # =================================================
            st.subheader("Main Features Driving Each Cluster")

            feature_df = get_cluster_features(result, X)

            top_n = st.slider(
                "Top Features Per Cluster",
                3, 10, 5
            )

            top_rows = []

            for c in sorted(feature_df["cluster"].unique()):
                top_rows.append(
                    feature_df[
                        feature_df["cluster"] == c
                    ].head(top_n)
                )

            st.dataframe(pd.concat(top_rows))

            # =================================================
            # DESCRIPTIVE CATEGORICAL CHARTS
            # =================================================
            st.subheader("Descriptive Cluster Profiles")

            for col_name in selected_cat_features:

                if col_name in X.columns:

                    xtab = pd.crosstab(
                        result["cluster"],
                        X[col_name],
                        normalize="index"
                    )

                    st.write(f"### {col_name}")

                    fig, ax = plt.subplots(figsize=(12, 6))

                    xtab.plot(
                        kind="bar",
                        stacked=True,
                        ax=ax
                    )

                    ax.set_title(
                        f"Distribution of {col_name} by Cluster"
                    )

                    ax.set_xlabel("Cluster")
                    ax.set_ylabel("Proportion")

                    plt.xticks(rotation=0)
                    plt.tight_layout()

                    st.pyplot(fig)

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

            if viz == "Cluster Counts":

                counts = (
                    result["cluster"]
                    .value_counts()
                    .sort_index()
                )

                st.bar_chart(counts)

            elif viz == "Cluster Map":

                fig2, ax2 = plt.subplots(figsize=(8, 5))

                ax2.scatter(
                    coords.iloc[:, 0],
                    coords.iloc[:, 1],
                    c=labels
                )

                ax2.set_xlabel("Dimension 1")
                ax2.set_ylabel("Dimension 2")
                ax2.set_title("Cluster Projection")

                st.pyplot(fig2)

            elif viz == "City by Cluster":

                if "City" in result.columns:

                    city_xtab = pd.crosstab(
                        result["City"],
                        result["cluster"]
                    )

                    st.dataframe(city_xtab)
                    st.bar_chart(city_xtab)

            elif viz == "Cluster Members":

                chosen = st.selectbox(
                    "Choose Cluster",
                    sorted(result["cluster"].unique())
                )

                st.dataframe(
                    result[result["cluster"] == chosen]
                )

            # =================================================
            # FINAL TABLE
            # =================================================
            st.subheader("Company ID Details and Cluster Allocation")
            st.dataframe(result)

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
