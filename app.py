

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

# Identifier columns removed from backend only
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
    "Upload a CSV dataset to automatically cluster SMEs. "
    "Identifier columns are excluded from modeling and returned in outputs."
)

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =====================================================
# MAIN APP
# =====================================================
if uploaded_file is not None:

    try:
        # ---------------------------------------------
        # LOAD DATA
        # ---------------------------------------------
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        # ---------------------------------------------
        # KEEP IDS THAT EXIST
        # ---------------------------------------------
        existing_ids = [col for col in id_cols if col in df.columns]

        ids = df[existing_ids].copy()

        # Remove IDs from backend
        X = df.drop(columns=existing_ids, errors="ignore").copy()

        # ---------------------------------------------
        # RUN ANALYSIS
        # ---------------------------------------------
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

            # -----------------------------------------
            # REATTACH IDS
            # -----------------------------------------
            result = pd.concat(
                [
                    ids.reset_index(drop=True),
                    pd.Series(labels, name="cluster")
                ],
                axis=1
            )

            # -----------------------------------------
            # RESULTS TABLE
            # -----------------------------------------
            st.subheader("Cluster Results")
            st.dataframe(result)

            # -----------------------------------------
            # VISUAL OPTIONS
            # -----------------------------------------
            st.subheader("Visualisations")

            viz = st.selectbox(
                "Choose visualisation",
                [
                    "Cluster Counts",
                    "Cluster Scatter Plot",
                    "City by Cluster",
                    "Cluster Members"
                ]
            )

            # -----------------------------------------
            # CLUSTER COUNTS
            # -----------------------------------------
            if viz == "Cluster Counts":

                counts = result["cluster"].value_counts().sort_index()
                st.bar_chart(counts)

            # -----------------------------------------
            # SCATTER PLOT
            # -----------------------------------------
            elif viz == "Cluster Scatter Plot":

                fig, ax = plt.subplots(figsize=(8, 5))

                ax.scatter(
                    X_famd.iloc[:, 0],
                    X_famd.iloc[:, 1],
                    c=labels
                )

                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.set_title("SME Cluster Map")

                st.pyplot(fig)

            # -----------------------------------------
            # CITY BY CLUSTER
            # -----------------------------------------
            elif viz == "City by Cluster":

                if "City" in result.columns:

                    city_table = pd.crosstab(
                        result["City"],
                        result["cluster"]
                    )

                    st.dataframe(city_table)
                    st.bar_chart(city_table)

                else:
                    st.warning("City column not found.")

            # -----------------------------------------
            # VIEW CLUSTER MEMBERS
            # -----------------------------------------
            elif viz == "Cluster Members":

                selected_cluster = st.selectbox(
                    "Select Cluster",
                    sorted(result["cluster"].unique())
                )

                members = result[
                    result["cluster"] == selected_cluster
                ]

                st.dataframe(members)

            # -----------------------------------------
            # DOWNLOAD
            # -----------------------------------------
            csv = result.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="cluster_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Upload a CSV file to begin.")
