import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import prince

from sklearn.cluster import KMeans

# =====================================
# CONFIG (HIDDEN FROM UI)
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
# PAGE
# =====================================
st.set_page_config(page_title="SME Clustering App", layout="wide")

st.title("SME Clustering App")
st.caption("Upload CSV data to automatically generate SME clusters.")

# =====================================
# HELPERS
# =====================================
def clean_columns(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("’", "'")
    )
    return df


def preprocess(df):
    df = df.copy()

    for col in df.columns:

        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({
                    "": "Unknown",
                    "nan": "Unknown",
                    "None": "Unknown"
                })
                .fillna("Unknown")
            )

        else:
            df[col] = df[col].fillna(0)

    return df

# =====================================
# UPLOAD
# =====================================
uploaded = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded is not None:

    df = pd.read_csv(uploaded)
    df = clean_columns(df)

    st.subheader("Preview")
    st.dataframe(df.head())

    # ---------------------------------
    # KEEP IDS
    # ---------------------------------
    existing_ids = [c for c in id_cols if c in df.columns]

    ids = df[existing_ids].copy()

    # ---------------------------------
    # REMOVE IDS FROM MODEL BACKEND
    # ---------------------------------
    X = df.drop(columns=existing_ids, errors="ignore")

    # ---------------------------------
    # PREPROCESS
    # ---------------------------------
    X = preprocess(X)

    # ---------------------------------
    # RUN MODEL
    # ---------------------------------
    if st.button("Run Analysis"):

        with st.spinner("Running clustering model..."):

            famd = prince.FAMD(
                n_components=N_COMPONENTS,
                random_state=42
            )

            X_famd = famd.fit_transform(X)

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
        # RESULTS
        # ---------------------------------
        st.subheader("Cluster Results")
        st.dataframe(result)

        st.subheader("Cluster Counts")
        st.bar_chart(result["cluster"].value_counts().sort_index())

        # ---------------------------------
        # CLUSTER MAP
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
        ax.set_title("SME Cluster Map")

        st.pyplot(fig)

        # ---------------------------------
        # DOWNLOAD
        # ---------------------------------
        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results CSV",
            csv,
            "cluster_results.csv",
            "text/csv"
        )

else:
    st.info("Upload a CSV file to begin.")
