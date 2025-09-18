# app.py
"""
Dashboard Desa (Lengkap)
Fitur utama:
- Load dataset parquet (tidak menghapus baris)
- Preprocess: redefine kemudahan dari jarak, mapping ke numeric
- Sidebar hierarchical filter (prov -> kab -> kec -> desa -> topografi)
- Tabs: Overview | Pendidikan | Kesehatan | Machine Learning
- Overview: summary + skor kelayakan + Top 3 fasilitas pendidikan dan kesehatan
- Pendidikan: Availability & Accessibility (histogram)
- Kesehatan: Availability, Accessibility, Tenaga, Fasilitas Khusus
- ML: KMeans clustering per kecamatan (menggunakan seluruh dataset, bukan filter)
- Label mapping supaya nama kolom tampil rapi
- Banyak caption & komentar agar mudah dipresentasikan
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from typing import Dict, Tuple, List

st.set_page_config(page_title="Dashboard Desa (Lengkap)", layout="wide")

# -----------------------------
# LABELS & KEYWORD CONFIG
# -----------------------------
# Human-readable labels for select columns (tambah sesuai kebutuhan)
LABELS = {
    # Pendidikan
    "paud_negeri": "PAUD Negeri", "paud_swasta": "PAUD Swasta",
    "tk_negeri": "TK Negeri", "tk_swasta": "TK Swasta",
    "raba_negeri": "RA/BA Negeri", "raba_swasta": "RA/BA Swasta",
    "sd_negeri": "SD Negeri", "sd_swasta": "SD Swasta",
    "mi_negeri": "MI Negeri", "mi_swasta": "MI Swasta",
    "smp_negeri": "SMP Negeri", "smp_swasta": "SMP Swasta",
    "mts_negeri": "MTS Negeri", "mts_swasta": "MTS Swasta",
    "sma_negeri": "SMA Negeri", "sma_swasta": "SMA Swasta",
    "ma_negeri": "MA Negeri", "ma_swasta": "MA Swasta",
    "smk_negeri": "SMK Negeri", "smk_swasta": "SMK Swasta",
    "perguruan_tinggi_negeri": "Perguruan Tinggi Negeri",
    "perguruan_tinggi_swasta": "Perguruan Tinggi Swasta",
    "pondok_pesantren_negeri": "Pondok Pesantren Negeri",
    "pondok_pesantren_swasta": "Pondok Pesantren Swasta",
    "madrasah_diniyah": "Madrasah Diniyah", "seminari": "Seminari",
    # SDLB/SMPLB/SMALB
    "sdlb_negeri": "SDLB Negeri", "sdlb_swasta": "SDLB Swasta",
    "smplb_negeri": "SMPLB Negeri", "smplb_swasta": "SMPLB Swasta",
    "smalb_negeri": "SMALB Negeri", "smalb_swasta": "SMALB Swasta",

    # Kesehatan
    "rumah_sakit": "Rumah Sakit",
    "rumah_sakit_bersalin": "Rumah Sakit Bersalin",
    "puskesmas_rawat_inap": "Puskesmas Rawat Inap",
    "puskesmas_pembantu": "Puskesmas Pembantu",
    "poliklinikbalai_pengobatan": "Poliklinik / Balai Pengobatan",
    "tempat_praktek_dokter": "Praktek Dokter",
    "rumah_bersalin": "Rumah Bersalin",
    "tempat_praktek_bidan": "Praktek Bidan",
    "poskesdes_pos_kesehatan_desa": "Poskesdes",
    "polindes_pondok_bersalin_desa": "Polindes",
    "apotek": "Apotek",
    "toko_khusus_obatjamu": "Toko Obat/Jamu",

    # Posyandu / kegiatan
    "posyandu_aktif": "Posyandu Aktif",
    "posyandu_kegiatanpelayanan_setiap_sebulan_sekali": "Posyandu (Bulanan)",
    "posyandu_kegiatanpelayanan_setiap_2_bulan_sekali_lebih": "Posyandu (2-bulanan+)",
    "pos_pembinaan_terpadu_posbindu": "Posbindu",
    "kader_pelaksana_kbkesehatan_ibu_anak": "Kader KIA",
    "warga_penderita_gizi_buruk": "Warga Penderita Gizi Buruk",

    # Tenaga kesehatan
    "tenaga_dokter_pria_tinggalmenetap_desakelurahan": "Dokter Pria (Menetap)",
    "tenaga_dokter_wanita_tinggalmenetap_desakelurahan": "Dokter Wanita (Menetap)",
    "tenaga_dokter_gigi_tinggalmenetap_desakelurahan": "Dokter Gigi (Menetap)",
    "tenaga_bidan_tinggalmenetap_desakelurahan": "Bidan (Menetap)",
    "tenaga_kesehatan_lain_tinggalmenetap_desakelurahan": "Tenaga Kesehatan Lain (Menetap)",

    # Contoh jarak labels
    "jarak_mencapai_paud": "Jarak ke PAUD (km)",
    "jarak_tk": "Jarak ke TK (km)",
    "jarak_raba": "Jarak ke RA/BA (km)",
    "jarak_sd": "Jarak ke SD (km)",
    "jarak_mi": "Jarak ke MI (km)",
    "jarak_smp": "Jarak ke SMP (km)",
    "jarak_mts": "Jarak ke MTS (km)",
    "jarak_sma": "Jarak ke SMA (km)",
    "jarak_ma": "Jarak ke MA (km)",
    "jarak_smk": "Jarak ke SMK (km)",
    "jarak_menuju_rumah_sakit": "Jarak ke Rumah Sakit (km)",
    "jarak_menuju_puskesmas_rawat_inap": "Jarak ke Puskesmas (km)",
    "jarak_menuju_apotek": "Jarak ke Apotek (km)",
}

# explicit keyword lists to select kemudahan_* columns (kehilangan leakage)
PEND_KEYWORDS = [
    "paud", "tk", "raba", "sd", "mi", "smp", "mts", "sma", "ma", "smk",
    "perguruan_tinggi", "sdlb", "smplb", "smalb", "pondok_pesantren", "madrasah_diniyah", "seminari"
]
KES_KEYWORDS = [
    "rumah_sakit", "rumah_sakit_bersalin", "puskesmas", "poliklinik", "tempat_praktek_dokter",
    "rumah_bersalin", "tempat_praktek_bidan", "poskesdes_pos_kesehatan_desa", "polindes_pondok_bersalin_desa",
    "apotek", "toko_khusus_obatjamu", "posyandu"
]
TENAGA_COLS = [
    "tenaga_dokter_pria_tinggalmenetap_desakelurahan",
    "tenaga_dokter_wanita_tinggalmenetap_desakelurahan",
    "tenaga_dokter_gigi_tinggalmenetap_desakelurahan",
    "tenaga_bidan_tinggalmenetap_desakelurahan",
    "tenaga_kesehatan_lain_tinggalmenetap_desakelurahan",
]
SPECIAL_HEALTH = [
    "posyandu_aktif",
    "posyandu_kegiatanpelayanan_setiap_sebulan_sekali",
    "posyandu_kegiatanpelayanan_setiap_2_bulan_sekali_lebih",
    "pos_pembinaan_terpadu_posbindu",
    "kader_pelaksana_kbkesehatan_ibu_anak",
    "warga_penderita_gizi_buruk"
]

# -----------------------------
# HELPERS
# -----------------------------
def label(col: str) -> str:
    """Return friendly label for a column if exists; else prettify the column name."""
    return LABELS.get(col, col.replace("_", " ").title())

# -----------------------------
# DATA LOADER & PREPROCESS
# -----------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load dataset parquet. Ubah path jika perlu."""
    # default path is dataset/dataset.parquet in project root
    return pd.read_parquet("dataset/dataset.parquet")


@st.cache_data
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess dataset:
    - Convert jarak_* to numeric (preserve NaN; no dropping)
    - Redefine kemudahan_mencapai_* based on jarak_* (resolves 'tidakberlaku' ambiguity)
    - Map kemudahan categories to numeric columns *_num
    - Return processed df and metadata dict with lists of kemudahan/jarak cols
    """
    df = df.copy()
    meta: Dict[str, List[str]] = {"kemudahan_cols": [], "jarak_cols": [], "kemudahan_num": [], "jarak_num": []}

    # detect jarak columns
    jarak_cols = [c for c in df.columns if c.startswith("jarak_")]
    for c in jarak_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # keep NaN (no drop)

    # mapping from distance to kemudahan category (heuristic)
    def map_kemudahan(jar):
        if pd.isna(jar):
            return "tidakberlaku"
        elif jar <= 5:
            return "sangatmudah"
        elif jar <= 10:
            return "mudah"
        elif jar <= 20:
            return "sulit"
        else:
            return "sangatsulit"

    # find kemudahan columns and attempt to redefine from jarak if possible
    kem_cols = [c for c in df.columns if c.startswith("kemudahan_mencapai_")]
    for jar_col in jarak_cols:
        kem_col = jar_col.replace("jarak_menuju_", "kemudahan_mencapai_").replace("jarak_", "kemudahan_mencapai_")
        if kem_col in df.columns:
            df[kem_col] = df[jar_col].apply(map_kemudahan)

    # mapping kemudahan to numeric 0..4
    mapping = {"sangatmudah": 4, "mudah": 3, "sulit": 2, "sangatsulit": 1, "tidakberlaku": 0}
    for c in kem_cols:
        df[c] = df[c].astype(str).str.lower().str.replace(r"\s+", "", regex=True)
        num_col = f"{c}_num"
        df[num_col] = df[c].map(mapping).fillna(0).astype(float)
        meta["kemudahan_cols"].append(c)
        meta["kemudahan_num"].append(num_col)

    meta["jarak_cols"] = jarak_cols
    meta["jarak_num"] = jarak_cols  # jarak already numeric

    return df, meta

# -----------------------------
# SIDEBAR FILTER (hierarki)
# -----------------------------
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build hierarchical filters (dependent dropdowns): prov -> kab -> kec -> desa -> topografi.
    The dropdown options are derived from current df (so they are dependent).
    """
    with st.sidebar:
        st.header("🔍 Filter Data (hierarki)")
        provs = ["All"] + sorted(df["nama_prov"].dropna().unique().tolist())
        prov_choice = st.selectbox("Pilih Provinsi", provs)
        if prov_choice != "All":
            df = df[df["nama_prov"] == prov_choice]

        kabs = ["All"] + sorted(df["nama_kab"].dropna().unique().tolist())
        kab_choice = st.selectbox("Pilih Kabupaten", kabs)
        if kab_choice != "All":
            df = df[df["nama_kab"] == kab_choice]

        kecs = ["All"] + sorted(df["nama_kec"].dropna().unique().tolist())
        kec_choice = st.selectbox("Pilih Kecamatan", kecs)
        if kec_choice != "All":
            df = df[df["nama_kec"] == kec_choice]

        desas = ["All"] + sorted(df["nama_desa"].dropna().unique().tolist())
        desa_choice = st.selectbox("Pilih Desa", desas)
        if desa_choice != "All":
            df = df[df["nama_desa"] == desa_choice]

        tops = ["All"] + sorted(df["topografi_wilayah_desakelurahan"].dropna().unique().tolist())
        topo_choice = st.selectbox("Pilih Topografi", tops)
        if topo_choice != "All":
            df = df[df["topografi_wilayah_desakelurahan"] == topo_choice]

    return df

# -----------------------------
# OVERVIEW (detailed) + TOP3 facilities
# -----------------------------
def show_overview(df: pd.DataFrame, meta: Dict):
    """
    Overview shows:
    - counts (prov/kab/kec/total rows)
    - indeks kelayakan (adaptive)
    - Top 3 fasilitas pendidikan & kesehatan (berdasarkan rata-rata skor kemudahan numeric)
    - jenis permukaan jalan per provinsi (split jika banyak provinsi)
    """
    st.header("📊 Overview")
    st.caption("Ringkasan dataset. Semua baris dipertahankan (tidak ada drop).")

    # basic counts
    total_rows = len(df)
    prov_count = df["nama_prov"].nunique()
    kab_count = df["nama_kab"].nunique()
    kec_count = df["nama_kec"].nunique()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Provinsi (unik)", prov_count)
    col2.metric("Kabupaten (unik)", kab_count)
    col3.metric("Kecamatan (unik)", kec_count)
    col4.metric("Desa (total baris)", total_rows)
    st.caption("Desa dihitung berdasarkan baris (mis. kalau 1 desa punya banyak fasilitas, dihitung beberapa baris).")

    # skor kelayakan adaptif
    st.subheader("Indeks Kelayakan (sederhana & adaptif)")
    kem_cols = meta.get("kemudahan_cols", [])
    if kem_cols:
        df["_skor_desa"] = df[kem_cols].apply(
            lambda row: float(np.mean([
                str(v).strip().lower() == "sangatmudah"
                for v in row if str(v).strip().lower() != "nan"
            ])) if any(pd.notna(row)) else 0.0,
            axis=1
        )
    else:
        df["_skor_desa"] = 0.0

    skor_mean = df["_skor_desa"].mean()
    st.metric("Skor Nasional (rata-rata proporsi 'sangatmudah')", f"{skor_mean:.3f}")
    st.caption("Skor desa = proporsi kemudahan yang berlabel 'sangatmudah' pada baris tersebut.")

    # adaptive grouping level (provinsi > kabupaten > kecamatan)
    if df["nama_prov"].nunique() > 1:
        group_col, group_label = "nama_prov", "Provinsi"
    elif df["nama_kab"].nunique() > 1:
        group_col, group_label = "nama_kab", "Kabupaten"
    else:
        group_col, group_label = "nama_kec", "Kecamatan"

    ranking = df.groupby(group_col)["_skor_desa"].mean().sort_values(ascending=False)
    if not ranking.empty:
        best = ranking.index[0]
        worst = ranking.index[-1]
        bcol, wcol = st.columns(2)
        bcol.metric(f"{group_label} Terbaik", str(best), f"{ranking.iloc[0]:.3f}")
        wcol.metric(f"{group_label} Terburuk", str(worst), f"{ranking.iloc[-1]:.3f}")
        fig = px.bar(ranking.head(10)[::-1], orientation="h",
                     labels={"x": "Skor (rata-rata proporsi sangatmudah)", "index": group_col},
                     title=f"Top 10 {group_label} (Skor Kelayakan)")
        st.plotly_chart(fig, use_container_width=True, key="overview_ranking")
        st.caption("Interpretasi: semakin mendekati 1 berarti proporsi fasilitas 'sangatmudah' lebih banyak → lebih layak.")

    # Top 3 Facilities — Pendidikan & Kesehatan
    st.subheader("Top 3 Fasilitas — Pendidikan & Kesehatan (berdasarkan skor rata-rata kemudahan numeric)")
    # compute mean per kemudahan_num column
    kem_num_cols = [c for c in meta.get("kemudahan_num", []) if c in df.columns]
    # derive mapping from kemudahan_mencapai_*_num back to base name
    if kem_num_cols:
        # mapping: base -> mean score
        facility_scores = {}
        for num_col in kem_num_cols:
            # num_col example: kemudahan_mencapai_paud_num -> base 'paud'
            base = num_col.replace("kemudahan_mencapai_", "").replace("_num", "")
            mean_score = df[num_col].mean()
            facility_scores[base] = mean_score

        # split into pend & kes using keywords
        pend_scores = {k: v for k, v in facility_scores.items() if any(pk in k for pk in PEND_KEYWORDS)}
        kes_scores = {k: v for k, v in facility_scores.items() if any(kk in k for kk in KES_KEYWORDS)}

        # get top3
        def top_n_dict(d: Dict[str, float], n=3):
            items = sorted(d.items(), key=lambda x: x[1], reverse=True)
            return items[:n]

        top3_pend = top_n_dict(pend_scores, 3)
        top3_kes = top_n_dict(kes_scores, 3)

        # display
        colp, colk = st.columns(2)
        with colp:
            st.markdown("**Top 3 (Pendidikan)**")
            if top3_pend:
                for i, (fac, sc) in enumerate(top3_pend, start=1):
                    pretty = label(f"kemudahan_mencapai_{fac}")
                    st.write(f"{i}. {pretty} — Rata-rata skor numeric: {sc:.3f}")
            else:
                st.write("Tidak ada data fasilitas pendidikan yang cukup.")
        with colk:
            st.markdown("**Top 3 (Kesehatan)**")
            if top3_kes:
                for i, (fac, sc) in enumerate(top3_kes, start=1):
                    pretty = label(f"kemudahan_mencapai_{fac}")
                    st.write(f"{i}. {pretty} — Rata-rata skor numeric: {sc:.3f}")
            else:
                st.write("Tidak ada data fasilitas kesehatan yang cukup.")
        st.caption("Catatan: skor numeric 4 = 'sangatmudah', 0 = 'tidakberlaku'. Top3 berarti rata-rata skor numeric tertinggi.")
    else:
        st.info("Tidak ada kolom kemudahan numeric untuk menghitung Top 3 fasilitas.")

    # jenis permukaan jalan (simple)
    st.subheader("Jenis Permukaan Jalan per Provinsi")
    if "jenis_permukaan_jalan" in df.columns:
        jalan = df.groupby(["nama_prov", "jenis_permukaan_jalan"]).size().reset_index(name="jumlah")
        nprov = jalan["nama_prov"].nunique()
        if nprov <= 20:
            fig = px.bar(jalan, x="nama_prov", y="jumlah", color="jenis_permukaan_jalan",
                         title="Jenis Permukaan Jalan per Provinsi")
            st.plotly_chart(fig, use_container_width=True, key="overview_jalan_all")
        else:
            provs = sorted(jalan["nama_prov"].unique())
            mid = len(provs) // 2
            for i, subset in enumerate([provs[:mid], provs[mid:]], start=1):
                jalan_sub = jalan[jalan["nama_prov"].isin(subset)]
                fig = px.bar(jalan_sub, x="nama_prov", y="jumlah", color="jenis_permukaan_jalan",
                             title=f"Jenis Permukaan Jalan (Bagian {i})")
                st.plotly_chart(fig, use_container_width=True, key=f"overview_jalan_part_{i}")
        st.caption("Jika proporsi 'AspalBeton' tinggi → infrastruktur lebih baik; jika 'Tanah' tinggi → tantangan akses lebih besar.")
    else:
        st.info("Kolom 'jenis_permukaan_jalan' tidak ditemukan.")

# -----------------------------
# PENDIDIKAN (detailed)
# -----------------------------
def show_pendidikan(df: pd.DataFrame, meta: Dict):
    """
    Pendidikan:
    - Availability: bar chart kategori kemudahan per fasilitas pendidikan
    - Accessibility: histogram jarak per fasilitas pendidikan (jika tersedia)
    """
    st.header("🎓 Pendidikan")
    st.caption("Analisis ketersediaan & aksesibilitas fasilitas pendidikan.")

    kem_cols = meta.get("kemudahan_cols", [])
    pend_cols = [c for c in kem_cols if any(k in c for k in PEND_KEYWORDS) and not any(k in c for k in KES_KEYWORDS)]

    if not pend_cols:
        st.info("Tidak ditemukan kolom kemudahan pendidikan pada subset ini.")
        return

    hide_tidak = st.checkbox("Sembunyikan kategori 'tidakberlaku' pada Availability (Pendidikan)", key="hide_pend_avail")

    tab_avail, tab_access = st.tabs(["Availability (Ketersediaan)", "Accessibility (Jarak)"])
    with tab_avail:
        st.subheader("Availability — Pendidikan")
        st.caption("Grafik ini menghitung jumlah baris per kategori kemudahan (bukan jumlah desa unik).")
        for i, kem_col in enumerate(pend_cols):
            counts = df[kem_col].value_counts().sort_index()
            if hide_tidak and "tidakberlaku" in counts.index:
                counts = counts.drop("tidakberlaku")
            x_labels = [f"{label(kem_col)} — {k}" for k in counts.index]
            color_map = {k: ("lightgrey" if k == "tidakberlaku" else None) for k in counts.index}
            fig = px.bar(x=x_labels, y=counts.values, text=counts.values,
                         labels={"x": "Kategori", "y": "Jumlah baris (desa/fasilitas)"},
                         title=f"Availability — {label(kem_col)}",
                         color=counts.index, color_discrete_map=color_map)
            st.plotly_chart(fig, use_container_width=True, key=f"pend_avail_{i}")
            st.caption("Interpretasi: nilai 'tidakberlaku' berarti fasilitas tidak terdaftar pada baris tersebut.")

    with tab_access:
        st.subheader("Accessibility — Pendidikan (Distribusi Jarak)")
        st.caption("Histogram jarak memberi gambaran sebaran jarak ke fasilitas (berguna meski fasilitas tidak ada di desa tersebut).")
        for i, kem_col in enumerate(pend_cols):
            base = kem_col.replace("kemudahan_mencapai_", "")
            candidates = [f"jarak_{base}", f"jarak_mencapai_{base}", f"jarak_menuju_{base}"]
            jar_col = next((c for c in candidates if c in df.columns), None)
            if jar_col is None:
                continue
            fig = px.histogram(df, x=jar_col, nbins=30, labels={jar_col: label(jar_col)},
                               title=f"Distribusi Jarak — {label(jar_col)}")
            st.plotly_chart(fig, use_container_width=True, key=f"pend_access_{i}")
            st.caption("Cara baca: puncak di sebelah kiri = banyak desa dekat; ekor kanan = beberapa desa jauh.")

# -----------------------------
# KESEHATAN (detailed: facilities, tenaga, special)
# -----------------------------
def show_kesehatan(df: pd.DataFrame, meta: Dict):
    """
    Kesehatan terbagi menjadi:
    - Fasilitas (Availability & Accessibility)
    - Tenaga Kesehatan (jumlah tenaga menetap)
    - Fasilitas Khusus / Kegiatan (posyandu, posbindu, gizi)
    """
    st.header("🏥 Kesehatan")
    st.caption("Analisis fasilitas & tenaga kesehatan serta fasilitas non-jarak (kegiatan).")

    kem_cols = meta.get("kemudahan_cols", [])
    kes_cols = [c for c in kem_cols if any(k in c for k in KES_KEYWORDS) and not any(k in c for k in PEND_KEYWORDS)]
    tenaga_present = [c for c in TENAGA_COLS if c in df.columns]
    special_present = [c for c in SPECIAL_HEALTH if c in df.columns]

    tab_fas, tab_tenaga, tab_special = st.tabs(["Fasilitas (Availability & Jarak)", "Tenaga Kesehatan", "Fasilitas Khusus"])

    # Fasilitas
    with tab_fas:
        st.subheader("Fasilitas Kesehatan — Availability")
        hide_tidak = st.checkbox("Sembunyikan kategori 'tidakberlaku' pada Availability (Kesehatan)", key="hide_kes_avail")
        if not kes_cols:
            st.info("Tidak ditemukan kolom kemudahan fasilitas kesehatan pada subset ini.")
        else:
            for i, kem_col in enumerate(kes_cols):
                counts = df[kem_col].value_counts().sort_index()
                if hide_tidak and "tidakberlaku" in counts.index:
                    counts = counts.drop("tidakberlaku")
                x_labels = [f"{label(kem_col)} — {k}" for k in counts.index]
                color_map = {k: ("lightgrey" if k == "tidakberlaku" else None) for k in counts.index}
                fig = px.bar(x=x_labels, y=counts.values, text=counts.values,
                             labels={"x": "Kategori", "y": "Jumlah baris (desa/fasilitas)"},
                             title=f"Availability — {label(kem_col)}",
                             color=counts.index, color_discrete_map=color_map)
                st.plotly_chart(fig, use_container_width=True, key=f"kes_avail_{i}")
                st.caption("Catatan: 'tidakberlaku' = tidak tercatat. Gunakan tab Jarak untuk melihat akses antardesa.")

        st.subheader("Fasilitas Kesehatan — Accessibility (Distribusi Jarak)")
        for i, kem_col in enumerate(kes_cols):
            base = kem_col.replace("kemudahan_mencapai_", "")
            candidates = [f"jarak_{base}", f"jarak_mencapai_{base}", f"jarak_menuju_{base}"]
            jar_col = next((c for c in candidates if c in df.columns), None)
            if jar_col is None:
                continue
            fig = px.histogram(df, x=jar_col, nbins=30, labels={jar_col: label(jar_col)},
                               title=f"Distribusi Jarak — {label(jar_col)}")
            st.plotly_chart(fig, use_container_width=True, key=f"kes_access_{i}")
            st.caption("Cara baca: perhatikan median & ekor distribusi untuk mengetahui apakah banyak desa jauh dari fasilitas.")

    # Tenaga Kesehatan
    with tab_tenaga:
        st.subheader("Ketersediaan Tenaga Kesehatan (Menetap)")
        if not tenaga_present:
            st.info("Kolom tenaga kesehatan tidak ditemukan di dataset.")
        else:
            agg = df[["nama_prov"] + tenaga_present].groupby("nama_prov").sum(min_count=1).fillna(0).reset_index()
            melt = agg.melt(id_vars="nama_prov", value_vars=tenaga_present, var_name="tenaga", value_name="jumlah")
            melt["tenaga_label"] = melt["tenaga"].map(lambda x: LABELS.get(x, x.replace("_", " ").title()))
            fig = px.bar(melt, x="nama_prov", y="jumlah", color="tenaga_label",
                         title="Jumlah Tenaga Kesehatan Menetap per Provinsi")
            st.plotly_chart(fig, use_container_width=True, key="kes_tenaga")
            st.caption("Jumlah tenaga menetap dijumlahkan per provinsi. Gunakan untuk memetakan distribusi personel kesehatan.")

            # totals table
            totals = melt.groupby("tenaga_label")["jumlah"].sum().sort_values(ascending=False)
            st.subheader("Ringkasan Total Tenaga Kesehatan (Semua Provinsi)")
            st.write(totals)
            st.caption("Tabel memberikan total tenaga per jenis (dokter, bidan, dll) di seluruh dataset.")

    # Fasilitas Khusus
    with tab_special:
        st.subheader("Fasilitas Khusus / Kegiatan (Posyandu, Posbindu, Gizi)")
        if not special_present:
            st.info("Tidak ditemukan kolom fasilitas khusus pada subset ini.")
        else:
            for i, col in enumerate(special_present):
                counts = df[col].value_counts().sort_index()
                x_labels = [f"{label(col)} — {k}" for k in counts.index]
                fig = px.bar(x=x_labels, y=counts.values, text=counts.values,
                             labels={"x": "Kategori", "y": "Jumlah baris"},
                             title=f"Distribusi — {label(col)}")
                st.plotly_chart(fig, use_container_width=True, key=f"kes_special_{i}")
                st.caption("Kolom ini menggambarkan kegiatan/keberadaan layanan di tingkat desa (bukan jarak).")

# -----------------------------
# MACHINE LEARNING (per kecamatan)
# -----------------------------
def show_ml(df: pd.DataFrame, meta: Dict):
    """
    Simple ML:
    - aggregate per kecamatan,
    - build skor_kelayakan (kombinasi kemudahan numeric & jarak normalisasi),
    - KMeans clustering (3 cluster).
    ML dijalankan pada keseluruhan dataset (bukan subset filter).
    """
    st.header("🤖 Machine Learning (Per Kecamatan)")
    st.caption("Analisis sederhana sebagai bahan prioritisasi: cluster kecamatan berdasarkan skor akses/kualitas fasilitas.")

    group_cols = ["nama_prov", "nama_kab", "nama_kec"]
    kem_num = meta.get("kemudahan_num", [])
    jar_cols = meta.get("jarak_num", [])

    use_kem = [c for c in kem_num if c in df.columns]
    use_jar = [c for c in jar_cols if c in df.columns]

    if not use_kem and not use_jar:
        st.info("Tidak ada kolom numerik untuk analisis ML.")
        return

    agg = df.groupby(group_cols)[use_kem + use_jar].mean()
    if agg.empty:
        st.warning("Agregasi per kecamatan kosong.")
        return

    # kem_score (0..4) -> kem_norm (0..1)
    if use_kem:
        kem_score = agg[use_kem].mean(axis=1)
    else:
        kem_score = pd.Series(0, index=agg.index)

    # jar_norm: normalize mean distance per kecamatan to 0..1
    if use_jar:
        jar_mean = agg[use_jar].replace(-1, np.nan).mean(axis=1)
        jar_mean = jar_mean.fillna(jar_mean.mean())  # fallback
        jar_norm = (jar_mean - jar_mean.min()) / (jar_mean.max() - jar_mean.min() + 1e-9)
    else:
        jar_norm = pd.Series(0, index=agg.index)

    kem_norm = kem_score / 4.0
    final_score = kem_norm * 0.7 + (1 - jar_norm) * 0.3
    agg["skor_kelayakan"] = final_score

    # KMeans clustering
    X = agg[["skor_kelayakan"]].fillna(0).values
    try:
        km = KMeans(n_clusters=3, random_state=42)
        labels = km.fit_predict(X)
    except Exception:
        labels = np.zeros(len(X), dtype=int)
    agg["cluster"] = labels

    plot_df = agg.reset_index().sort_values("skor_kelayakan", ascending=False).reset_index(drop=True)

    st.subheader("Top 10 Kecamatan (Skor Kelayakan Tertinggi)")
    top10 = plot_df.head(10)
    if not top10.empty:
        fig = px.bar(top10, x="nama_kec", y="skor_kelayakan", color="nama_prov", title="Top 10 Kecamatan (Skor Kelayakan)")
        st.plotly_chart(fig, use_container_width=True, key="ml_top10")
        st.caption("Top 10 kecamatan yang relatif paling layak berdasarkan skor gabungan kemudahan & jarak.")
    else:
        st.info("Tidak ada data untuk Top 10.")

    st.subheader("Cluster Kelayakan (3 kelompok)")
    fig2 = px.scatter(plot_df, x=plot_df.index, y="skor_kelayakan", color=plot_df["cluster"].astype(str),
                      hover_data=["nama_prov", "nama_kab", "nama_kec"],
                      labels={"x": "Index (urut by skor)", "y": "Skor Kelayakan"},
                      title="Cluster Kelayakan (3 kelompok)")
    st.plotly_chart(fig2, use_container_width=True, key="ml_cluster")
    st.caption("Cluster 0/1/2 memisahkan kecamatan menurut skor. Gunakan output tabel untuk melihat kecamatan per cluster.")

    st.write("Distribusi cluster (jumlah kecamatan per cluster):")
    st.write(plot_df["cluster"].value_counts().sort_index())

    if st.button("Export Top 10 Kecamatan ke CSV (ML)"):
        csv = top10.to_csv(index=False)
        st.download_button("Download top10_ml.csv", csv, "top10_kecamatan_ml.csv", "text/csv")

# -----------------------------
# MAIN
# -----------------------------
def main():
    st.title("📊 Dashboard Fasilitas Pendidikan & Kesehatan Desa — Versi Lengkap")
    st.write("Catatan: preprocessing tidak menghapus baris. Kemudahan didefinisikan ulang dari jarak bila tersedia.")

    # load & preprocess
    df = load_data()
    df, meta = preprocess_data(df)

    # apply filters (hierarki)
    df_filtered = filter_data(df)

    # tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Pendidikan", "Kesehatan", "Machine Learning"])
    with tab1:
        show_overview(df_filtered, meta)
    with tab2:
        show_pendidikan(df_filtered, meta)
    with tab3:
        show_kesehatan(df_filtered, meta)
    with tab4:
        # ML intentionally uses full dataset (not filtered)
        show_ml(df, meta)

if __name__ == "__main__":
    main()
