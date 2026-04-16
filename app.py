# app.py
# Lightweight SME Clustering App (1GB friendly)
# Includes cluster cross-tabs + requested visualisations

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
    "Low-memory clustering app with cluster visuals and cross-tabs."
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

# =====================================================
# MODEL FUNCTION (CACHED)
# =====================================================
@st.cache_data(show_spinner=False)
def run_model(df):

    # Row cap
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
    coords.columns = ["dim_0", "dim_1"]

    # Results
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
                f"Using first {MAX_ROWS} rows to stay within memory limits."
            )

        if st.button("Run Analysis"):

            with st.spinner("Running clustering..."):

                result, coords, labels, X = run_model(df)

            # =================================================
            # RESULTS
            # =================================================
            st.subheader("Cluster Results")
            st.dataframe(result)

            # =================================================
            # VISUAL MENU
            # =================================================
            st.subheader("Visualisations")

            viz = st.selectbox(
                "Choose visualisation",
                [
                    "Cluster Counts",
                    "Cluster Map",
                    "City by Cluster",
                    "Cluster Members",
                    "Cross-tab by Variable"
                ]
            )

            # ---------------------------------------------
            # 1. CLUSTER COUNTS
            # ---------------------------------------------
            if viz == "Cluster Counts":

                counts = (
                    result["cluster"]
                    .value_counts()
                    .sort_index()
                )

                st.bar_chart(counts)

            # ---------------------------------------------
            # 2. CLUSTER MAP
            # ---------------------------------------------
            elif viz == "Cluster Map":

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

            # ---------------------------------------------
            # 3. CITY BY CLUSTER
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
            # 4. CLUSTER MEMBERS
            # ---------------------------------------------
            elif viz == "Cluster Members":

                selected_cluster = st.selectbox(
                    "Choose Cluster",
                    sorted(result["cluster"].unique())
                )

                filtered = result[
                    result["cluster"] == selected_cluster
                ]

                st.dataframe(filtered)

            # ---------------------------------------------
            # 5. VARIABLE CROSS-TAB
            # ---------------------------------------------
            elif viz == "Cross-tab by Variable":

                cat_cols = X.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                if len(cat_cols) > 0:

                    chosen = st.selectbox(
                        "Choose variable",
                        cat_cols
                    )

                    xtab = pd.crosstab(
                        result["cluster"],
                        result[chosen],
                        normalize="index"
                    )

                    st.dataframe(xtab)

                    fig2, ax2 = plt.subplots(
                        figsize=(12, 6)
                    )

                    xtab.plot(
                        kind="bar",
                        stacked=True,
                        ax=ax2
                    )

                    ax2.set_title(
                        f"{chosen} by Cluster"
                    )

                    ax2.set_xlabel("Cluster")
                    ax2.set_ylabel("Proportion")

                    plt.xticks(rotation=0)
                    plt.tight_layout()

                    st.pyplot(fig2)

                else:
                    st.warning(
                        "No categorical columns available."
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
