# app.py
# DEBUG VERSION
# Shows exactly where startup/runtime breaks

import streamlit as st

# -------------------------------------------------
# EARLY PAGE LOAD
# -------------------------------------------------
st.set_page_config(page_title="SME Clustering App", layout="wide")
st.title("SME Clustering App - Debug Mode")

st.write("✅ App file started successfully")

# -------------------------------------------------
# DEBUG IMPORTS
# -------------------------------------------------
try:
    import pandas as pd
    st.write("✅ pandas imported")
except Exception as e:
    st.error(f"❌ pandas import failed: {e}")
    st.stop()

try:
    import matplotlib.pyplot as plt
    st.write("✅ matplotlib imported")
except Exception as e:
    st.error(f"❌ matplotlib import failed: {e}")
    st.stop()

try:
    import prince
    st.write("✅ prince imported")
except Exception as e:
    st.error(f"❌ prince import failed: {e}")
    st.stop()

try:
    from sklearn.cluster import KMeans
    st.write("✅ sklearn imported")
except Exception as e:
    st.error(f"❌ sklearn import failed: {e}")
    st.stop()

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
N_COMPONENTS = 3
N_CLUSTERS = 4

id_cols = [
    "Company name",
    "Company Email",
    "Company number",
    "City"
]

st.write("✅ Settings loaded")

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    try:
        st.write("✅ File uploaded")

        # -----------------------------------------
        # READ CSV
        # -----------------------------------------
        df = pd.read_csv(uploaded_file)
        st.write("✅ CSV read successfully")

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        st.dataframe(df.head())

        # -----------------------------------------
        # IDS
        # -----------------------------------------
        existing_ids = [c for c in id_cols if c in df.columns]
        st.write("✅ Existing ID columns:", existing_ids)

        ids = df[existing_ids].copy()

        # -----------------------------------------
        # REMOVE IDS
        # -----------------------------------------
        X = df.drop(columns=existing_ids, errors="ignore").copy()

        st.write("✅ Modeling dataset created")
        st.write("Shape:", X.shape)

        # -----------------------------------------
        # RUN BUTTON
        # -----------------------------------------
        if st.button("Run Analysis"):

            st.write("▶️ Starting model run")

            # -------------------------------------
            # FAMD INIT
            # -------------------------------------
            try:
                famd = prince.FAMD(
                    n_components=N_COMPONENTS,
                    random_state=42
                )
                st.write("✅ FAMD initialized")
            except Exception as e:
                st.error(f"❌ FAMD init failed: {e}")
                st.stop()

            # -------------------------------------
            # FAMD FIT
            # -------------------------------------
            try:
                X_famd = famd.fit_transform(X)
                st.write("✅ FAMD fit_transform complete")
            except Exception as e:
                st.error(f"❌ FAMD fit_transform failed: {e}")
                st.stop()

            # -------------------------------------
            # KMEANS INIT
            # -------------------------------------
            try:
                kmeans = KMeans(
                    n_clusters=N_CLUSTERS,
                    random_state=42,
                    n_init=10
                )
                st.write("✅ KMeans initialized")
            except Exception as e:
                st.error(f"❌ KMeans init failed: {e}")
                st.stop()

            # -------------------------------------
            # KMEANS FIT
            # -------------------------------------
            try:
                clusters = kmeans.fit_predict(X_famd)
                st.write("✅ KMeans clustering complete")
            except Exception as e:
                st.error(f"❌ KMeans fit failed: {e}")
                st.stop()

            # -------------------------------------
            # RESULTS
            # -------------------------------------
            try:
                X_famd_df = X_famd.copy()
                X_famd_df.columns = [
                    f"dim_{i}" for i in range(X_famd_df.shape[1])
                ]

                result_df = pd.concat(
                    [
                        ids.reset_index(drop=True),
                        X_famd_df.reset_index(drop=True),
                        pd.Series(clusters, name="cluster")
                    ],
                    axis=1
                )

                st.write("✅ Results table created")
                st.dataframe(result_df.head())

            except Exception as e:
                st.error(f"❌ Result creation failed: {e}")
                st.stop()

    except Exception as e:
        st.error(f"❌ CSV processing failed: {e}")

else:
    st.info("Upload a CSV file to begin.")
