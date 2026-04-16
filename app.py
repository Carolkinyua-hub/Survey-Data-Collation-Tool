# app.py
# Lightweight 1GB-safe version
# Reduced memory usage / reduced looping / cached run

import streamlit as st
import pandas as pd
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
    "Lightweight clustering app optimized for low-memory deployment."
)

# =====================================================
# FIXED SETTINGS
# =====================================================
N_COMPONENTS = 2          # reduced from 8
N_CLUSTERS = 4
MAX_ROWS = 500           # cap rows for memory

id_cols = [
    "Company name",
    "Company Email",
    "Company number",
    "City"
]

# =====================================================
# CACHED MODEL FUNCTION
# =====================================================
@st.cache_data(show_spinner=False)
def run_model(df):

    # Reduce rows for memory safety
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS).copy()

    # Keep only existing ID cols
    existing_ids = [c for c in id_cols if c in df.columns]

    ids = df[existing_ids].copy()

    # Remove IDs
    X = df.drop(columns=existing_ids, errors="ignore").copy()

    # Remove columns with only one unique value
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]

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

    # Coordinates
    coords = X_famd.copy()
    coords.columns = [
        "dim_0",
        "dim_1"
    ]

    # Final output
    result = pd.concat(
        [
            ids.reset_index(drop=True),
            pd.Series(labels, name="cluster")
        ],
        axis=1
    )

    return result, coords, labels

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
                f"Dataset capped to first {MAX_ROWS} rows for performance."
            )

        if st.button("Run Analysis"):

            with st.spinner("Running lightweight clustering..."):

                result, coords, labels = run_model(df)

            # -----------------------------------------
            # RESULTS
            # -----------------------------------------
            st.subheader("Cluster Results")
            st.dataframe(result)

            # -----------------------------------------
            # COUNTS
            # -----------------------------------------
            st.subheader("Cluster Counts")

            counts = (
                result["cluster"]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(counts)

            # -----------------------------------------
            # SCATTER
            # -----------------------------------------
            st.subheader("Cluster Map")

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.scatter(
                coords["dim_0"],
                coords["dim_1"],
                c=labels
            )

            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.set_title("FAMD Projection")

            st.pyplot(fig)

            # -----------------------------------------
            # FILTER
            # -----------------------------------------
            st.subheader("View Cluster Members")

            selected_cluster = st.selectbox(
                "Choose Cluster",
                sorted(result["cluster"].unique())
            )

            st.dataframe(
                result[
                    result["cluster"] == selected_cluster
                ]
            )

            # -----------------------------------------
            # DOWNLOAD
            # -----------------------------------------
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
        st.error(f"Error: {str(e)}")

else:
    st.info("Upload CSV file to begin.")
