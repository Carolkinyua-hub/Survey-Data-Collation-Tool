
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import prince
from sklearn.cluster import KMeans

# =====================================
# FIXED BACKEND SETTINGS
# =====================================
N_COMPONENTS = 8
N_CLUSTERS = 4

id_cols = [
    "Company name",
    "Company Email",
    "Company number",
    "City"
]

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="SME Clustering App",
    layout="wide"
)

st.title("SME Clustering App")
st.caption(
    "Upload a CSV file. Identifier columns are removed from clustering "
    "and returned in final results."
)


# =====================================
# FILE UPLOAD
# =====================================
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # ---------------------------------
    # LOAD DATA
    # ---------------------------------
    df = pd.read_csv(uploaded_file)
    df = clean_columns(df)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # ---------------------------------
    # IDENTIFY AVAILABLE ID COLUMNS
    # ---------------------------------
    existing_ids = [col for col in id_cols if col in df.columns]

    # Save IDs for later reattachment
    ids = df[existing_ids].copy()

    # Remove IDs from backend only
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
        # RETURN IDS AFTER ANALYSIS
        # ---------------------------------
        result = pd.concat(
            [
                ids.reset_index(drop=True),
                pd.Series(labels, name="cluster")
            ],
            axis=1
        )

        # ---------------------------------
        # RESULTS TABLE
        # ---------------------------------
        st.subheader("Cluster Results")
        st.dataframe(result)

        # ---------------------------------
        # CLUSTER COUNTS
        # ---------------------------------
        st.subheader("Cluster Counts")
        st.bar_chart(
            result["cluster"].value_counts().sort_index()
        )

        # ---------------------------------
        # VISUALIZATION
        # ---------------------------------
        st.subheader("Cluster Visualization")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(
            X_famd.iloc[:, 0],
            X_famd.iloc[:, 1],
            c=labels
        )

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title("FAMD SME Cluster Map")

        st.pyplot(fig)

        # ---------------------------------
        # FILTER BY CLUSTER
        # ---------------------------------
        st.subheader("View SMEs by Cluster")

        selected_cluster = st.selectbox(
            "Select Cluster",
            sorted(result["cluster"].unique())
        )

        st.dataframe(
            result[result["cluster"] == selected_cluster]
        )

        # ---------------------------------
        # DOWNLOAD
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
