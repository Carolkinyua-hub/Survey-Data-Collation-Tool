# app.py
# SME Clustering App
# Remove IDs before clustering -> reattach IDs after clustering
# User chooses visualizations after results

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import prince

from sklearn.cluster import KMeans

# =====================================================
# FIXED BACKEND SETTINGS
# =====================================================
N_COMPONENTS = 8
N_CLUSTERS = 4

# Columns treated as identifiers (removed from model)
id_cols = [
    "Company name",
    "Company Email",
    "Company number",
    "City"
]

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="SME Clustering App",
    layout="wide"
)

st.title("SME Clustering App")
st.caption(
    "Upload a CSV file. Identifier columns are excluded from clustering "
    "and returned in the final results."
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

    # ---------------------------------
    # LOAD DATA
    # ---------------------------------
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # ---------------------------------
    # KEEP ONLY IDS THAT EXIST
    # ---------------------------------
    existing_ids = [col for col in id_cols if col in df.columns]

    # Save IDs for later
    ids = df[existing_ids].copy()

    # Remove IDs from backend
    X = df.drop(columns=existing_ids, errors="ignore").copy()

    # ---------------------------------
    # RUN MODEL
    # ---------------------------------
    if st.button("Run Analysis"):

        with st.spinner("Running clustering analysis..."):

            # FAMD
            famd = prince.FAMD(
                n_components=N_COMPONENTS,
                random_state=42
            )

            X_famd = famd.fit_transform(X)

            # KMeans
            kmeans = KMeans(
                n_clusters=N_CLUSTERS,
                random_state=42,
                n_init=10
            )

            labels = kmeans.fit_predict(X_famd)

        # ---------------------------------
        # REATTACH IDS
        # ---------------------------------
        result = pd.concat(
            [
                ids.reset_index(drop=True),
                pd.Series(labels, name="cluster")
            ],
            axis=1
        )

        # ---------------------------------
        # SHOW RESULTS
        # ---------------------------------
        st.subheader("Cluster Results")
        st.dataframe(result)

        # ---------------------------------
        # VISUALIZATION CHOICE
        # ---------------------------------
        st.subheader("Choose Visualization")

        viz = st.selectbox(
            "Select Visualization",
            [
                "Cluster Counts",
                "Scatter Plot",
                "City by Cluster",
                "View Cluster Members"
            ]
        )

        # ---------------------------------
        # 1. CLUSTER COUNTS
        # ---------------------------------
        if viz == "Cluster Counts":

            counts = result["cluster"].value_counts().sort_index()

            st.bar_chart(counts)

        # ---------------------------------
        # 2. SCATTER PLOT
        # ---------------------------------
        elif viz == "Scatter Plot":

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

        # ---------------------------------
        # 3. CITY BY CLUSTER
        # ---------------------------------
        elif viz == "City by Cluster":

            if "City" in result.columns:

                city_table = pd.crosstab(
                    result["City"],
                    result["cluster"]
                )

                st.dataframe(city_table)

                st.bar_chart(city_table)

            else:
                st.warning("City column not found in uploaded dataset.")

        # ---------------------------------
        # 4. VIEW MEMBERS
        # ---------------------------------
        elif viz == "View Cluster Members":

            selected_cluster = st.selectbox(
                "Select Cluster",
                sorted(result["cluster"].unique())
            )

            filtered = result[
                result["cluster"] == selected_cluster
            ]

            st.dataframe(filtered)

        # ---------------------------------
        # DOWNLOAD RESULTS
        # ---------------------------------
        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="cluster_results.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a CSV file to begin.")
