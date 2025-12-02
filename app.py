import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Film Clustering Analysis", layout="wide", page_icon="🎬")
st.title("🎬 Analisis Segmentasi Film (Rating, Metascore, Gross)")

# Sidebar
uploaded_file = st.sidebar.file_uploader("Upload Dataset IMDb (.csv)", type=["csv"])
num_clusters = st.sidebar.slider("Pilih Jumlah Cluster (k)", 2, 10, 3)
run_cluster = st.sidebar.button("Jalankan K-Means")

# NAVBAR HORIZONTAL
menu = st.tabs([
    "📄 Dataset Awal",
    "📊 Perhitungan Clustering",
    "🎯 Visualisasi 2D",
    "🌀 Visualisasi 3D",
    "📈 Histogram Distribusi",
    "🔥 Heatmap Korelasi",
    "🌳 Dendrogram (Linkage)"
])

# money parser
def parse_money(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s.upper() in ["", "N/A", "-", "NONE"]: return np.nan
    s = s.replace(",", "").replace(" ", "")
    m = re.match(r'^\$?([0-9]*\.?[0-9]+)([MKk])?$', s)
    if m:
        val = float(m.group(1))
        suf = m.group(2)
        if suf:
            return val * (1_000_000 if suf.upper()=="M" else 1_000)
        return val
    digits = re.sub(r'[^0-9.]', '', s)
    try: return float(digits)
    except: return np.nan


# ======================================================
#               PROCESSING DATASET
# ======================================================
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    # cek kolom
    expected = ['rating', 'metascore', 'gross']
    missing = [c for c in expected if c not in df_raw.columns]
    if missing:
        st.error(f"Kolom hilang: {missing}")
        st.stop()

    df = df_raw.copy()
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
    df['gross_parsed'] = df['gross'].apply(parse_money)

    cols_main = ['rating', 'metascore', 'gross_parsed']
    df[cols_main] = df[cols_main].replace(0, np.nan)

    df_clean_nan = df.copy()
    df_cluster_ready = df_clean_nan.dropna(subset=cols_main).reset_index(drop=True)

    scaler = StandardScaler()
    X = df_cluster_ready[cols_main].astype(float).values
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=["rating", "metascore", "gross"])

    if run_cluster:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        df_cluster_ready["cluster"] = labels
        df_scaled["cluster"] = labels

else:
    st.warning("Silakan upload file terlebih dahulu.")
    st.stop()


# ======================================================
#                  NAVBAR CONTENT
# ======================================================

# ---------- DATASET AWAL ----------
with menu[0]:
    st.subheader("📌 Dataset Mentah (Preview)")
    st.dataframe(df_raw.head())

    st.subheader("📊 Missing Value")
    missing_info = df[cols_main].isna().sum().to_frame("missing")
    missing_info["percent"] = missing_info["missing"] / len(df) * 100
    st.dataframe(missing_info)


# ---------- PERHITUNGAN ----------
with menu[1]:
    st.subheader("📌 Setelah Drop Missing")
    st.write(f"Baris sebelum: {len(df)}, sesudah cleaning: {len(df_cluster_ready)}")
    st.dataframe(df_cluster_ready.head())

    if run_cluster:
        st.subheader("📋 Ringkasan Statistik Tiap Cluster (Skala Asli)")
        summary = df_cluster_ready.groupby("cluster")[cols_main].mean()
        st.dataframe(summary)

        csv = df_cluster_ready.to_csv(index=False).encode("utf-8")
        st.download_button("Download Hasil Clustering", csv, "cluster_output.csv")

        # ==========================
        #    KESIMPULAN CLUSTERING
        # ==========================
        st.subheader("📝 Kesimpulan Analisis Clustering")

        # Interpretasi otomatis berdasarkan ranking rata-rata
        cluster_summary = summary.copy()
        cluster_summary_rank = cluster_summary.rank(ascending=False)

        interpretation = []
        for c in cluster_summary.index:
            rating_lvl = cluster_summary_rank.loc[c, 'rating']
            meta_lvl = cluster_summary_rank.loc[c, 'metascore']
            gross_lvl = cluster_summary_rank.loc[c, 'gross_parsed']

            desc = f"**Cluster {c}** memiliki:"
            if rating_lvl == 1:
                desc += " ⭐ *Rating tertinggi*,"
            elif rating_lvl == num_clusters:
                desc += " ⬇️ *Rating terendah*,"
            
            if meta_lvl == 1:
                desc += " 🎭 *Metascore terbaik*,"
            elif meta_lvl == num_clusters:
                desc += " 💢 *Metascore paling rendah*,"

            if gross_lvl == 1:
                desc += " 💰 *Pendapatan (Gross) terbesar*,"
            elif gross_lvl == num_clusters:
                desc += " 📉 *Pendapatan (Gross) terendah*,"

            interpretation.append(desc.rstrip(","))

        for line in interpretation:
            st.markdown(f"- {line}")

        st.markdown("---")
        st.markdown("""
        **Kesimpulan Umum:**
        - Clustering membagi film menjadi beberapa kelompok berdasarkan **rating**, **metascore**, dan **gross**.
        - Setiap cluster menunjukkan karakteristik unik seperti:
            - film berkualitas tinggi,
            - film populer berpendapatan tinggi,
            - film ber-rating rendah namun pendapatan tinggi,
            - atau film yang performanya lemah di semua aspek.
        - Hasil clustering dapat membantu segmentasi film untuk analisis industri, rekomendasi, atau pemasaran.
        """)

    else:
        st.info("Klik 'Jalankan K-Means' untuk melihat hasil.")


# ---------- VISUALISASI 2D ----------
with menu[2]:
    if run_cluster:
        st.subheader("📊 Visualisasi 2D: Rating vs Metascore")
        fig, ax = plt.subplots(figsize=(7,5))
        scatter = ax.scatter(df_scaled["rating"], df_scaled["metascore"], 
                             c=df_scaled["cluster"], s=40)
        ax.set_xlabel("Rating (standardized)")
        ax.set_ylabel("Metascore (standardized)")
        ax.set_title("Rating vs Metascore")
        ax.legend(*scatter.legend_elements(), title="Cluster")
        st.pyplot(fig)

        st.subheader("📊 Visualisasi 2D: Rating vs Gross")
        fig2, ax2 = plt.subplots(figsize=(7,5))
        scatter2 = ax2.scatter(df_scaled["rating"], df_scaled["gross"], 
                               c=df_scaled["cluster"], s=40)
        ax2.set_xlabel("Rating (standardized)")
        ax2.set_ylabel("Gross (standardized)")
        ax2.set_title("Rating vs Gross")
        ax2.legend(*scatter2.legend_elements(), title="Cluster")
        st.pyplot(fig2)

    else:
        st.info("Jalankan K-Means terlebih dahulu.")


# ---------- VISUALISASI 3D ----------
with menu[3]:
    if run_cluster:
        st.subheader("📈 Visualisasi 3D")
        fig3d = plt.figure(figsize=(8,6))
        ax3d = fig3d.add_subplot(111, projection="3d")
        p = ax3d.scatter(df_scaled["rating"], df_scaled["metascore"], df_scaled["gross"],
                         c=df_scaled["cluster"], s=50)
        ax3d.set_xlabel("Rating")
        ax3d.set_ylabel("Metascore")
        ax3d.set_zlabel("Gross")
        st.pyplot(fig3d)
    else:
        st.info("Jalankan K-Means dulu.")


# ---------- HISTOGRAM DISTRIBUSI ----------
with menu[4]:
        st.subheader("📈 Histogram Distribusi Fitur")
        fig, ax = plt.subplots(figsize=(10,6))
        df_cluster_ready[['rating','metascore','gross_parsed']].hist(ax=ax, bins=30)
        plt.tight_layout()
        st.pyplot(fig)

# ---------- HEATMAP KORELASI ----------
with menu[5]:
    st.subheader("🔥 Heatmap Korelasi")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(df_cluster_ready[['rating','metascore','gross_parsed']].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------- DENDROGRAM LINKAGE ----------
with menu[6]:
    st.subheader("🌳 Dendrogram (Hierarchical Linkage)")
    Z = linkage(df_scaled[["rating","metascore","gross"]], method="ward")

    fig_den, axden = plt.subplots(figsize=(10,5))
    dendrogram(Z, ax=axden)
    axden.set_title("Dendrogram - Linkage Ward")
    st.pyplot(fig_den)
