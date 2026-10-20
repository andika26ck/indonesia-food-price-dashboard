import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
import json
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🇮🇩 Dapur Data Indonesia",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

    /* Background */
    .stApp {
        background-color: #FDF6EC;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1B4332;
    }
    [data-testid="stSidebar"] * {
        color: #D8F3DC !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] h1, h2, h3 {
        color: #D8F3DC !important;
    }

    /* Hero header */
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        color: #1B4332;
        line-height: 1.1;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #6B7280;
        margin-top: 0.3rem;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        border-left: 5px solid #2D6A4F;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 0.8rem;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #9CA3AF;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B4332;
        line-height: 1;
    }
    .metric-delta {
        font-size: 0.82rem;
        color: #6B7280;
        margin-top: 0.2rem;
    }

    /* AI Predictor card */
    .ai-card {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        margin-bottom: 1rem;
    }
    .ai-card h3 {
        color: white !important;
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }
    .ai-card p {
        color: #B7E4C7;
        font-size: 0.9rem;
    }

    /* Label badges */
    .badge-mahal {
        background: #FEE2E2; color: #991B1B;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-sedang {
        background: #FEF9C3; color: #854D0E;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-murah {
        background: #DCFCE7; color: #166534;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }

    /* Section titles */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #1B4332;
        border-bottom: 2px solid #B7E4C7;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Chart containers */
    .chart-card {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    /* AI response box */
    .ai-response {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        font-size: 0.95rem;
        color: #1B4332;
        line-height: 1.7;
    }

    /* Highlight number */
    .big-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1B4332;
    }

    div[data-testid="stTabs"] button[role="tab"] {
        font-weight: 600;
        color: #2D6A4F;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        border-bottom-color: #2D6A4F !important;
        color: #1B4332 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── LOAD DATA & MODELS ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_dashboard.csv")
    return df

@st.cache_resource
def load_models():
    with open("label_encoders.pkl", "rb") as f:
        le = pickle.load(f)
    scaler = joblib.load("scaler_harga.pkl")
    model = joblib.load("model_rf_harga.pkl")
    return le, scaler, model

df = load_data()
le, scaler, model = load_models()

PROVINCES = sorted([p for p in df["admin1"].unique() if p != "NATIONAL AVERAGE"])
COMMODITIES = sorted(df["commodity"].unique().tolist())
CATEGORIES = sorted(df["category"].unique().tolist())
YEARS = sorted(df["year"].unique().tolist())

COMMODITY_EMOJI = {
    "RICE": "🍚", "WHEAT FLOUR": "🌾", "EGGS": "🥚", "MEAT (BEEF)": "🥩",
    "MEAT (CHICKEN, BROILER)": "🍗", "MILK (CONDENSED)": "🥛", "SUGAR": "🍬",
    "FUEL (KEROSENE)": "⛽", "OIL (VEGETABLE)": "🫙", "CHILI (BIRD'S EYE)": "🌶️",
    "CHILI (RED)": "🌶️", "GARLIC": "🧄", "ONIONS (SHALLOT)": "🧅",
}


# ─── HELPER: PREDICT ────────────────────────────────────────────────────────────
def predict_label(year, month, province, commodity):
    try:
        row_df = pd.DataFrame([[year, month]], columns=["year", "month"])
        scaled = scaler.transform(row_df)

        cat_row = df[df["commodity"] == commodity]["category"].iloc[0]
        market_candidates = df[
            (df["admin1"] == province) & (df["commodity"] == commodity)
        ]["market"].unique()
        market = market_candidates[0] if len(market_candidates) > 0 else df[df["admin1"] == province]["market"].iloc[0]

        admin1_enc = le["admin1"].transform([province])[0]
        market_enc = le["market"].transform([market])[0]
        cat_enc = le["category"].transform([cat_row])[0]
        com_enc = le["commodity"].transform([commodity])[0]

        X = pd.DataFrame([[scaled[0][0], scaled[0][1], admin1_enc, market_enc, cat_enc, com_enc]],
                         columns=["year", "month", "admin1", "market", "category", "commodity"])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        conf = dict(zip(model.classes_, proba))
        return pred, conf
    except Exception as e:
        return None, {}


# ─── HELPER: CLAUDE ANALYSIS ────────────────────────────────────────────────────
def ask_claude(prompt, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍚 Dapur Data")
    st.markdown("**Indonesia Food Prices**")
    st.markdown("---")

    st.markdown("### 🔍 Filter Global")
    sel_province = st.selectbox("Provinsi", ["NATIONAL AVERAGE"] + PROVINCES)
    sel_year_range = st.slider("Rentang Tahun", 2007, 2023, (2015, 2023))
    sel_category = st.multiselect("Kategori", CATEGORIES, default=CATEGORIES[:3])

    st.markdown("---")
    st.markdown("### 🤖 Claude AI Key")
    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    st.caption("Diperlukan untuk fitur Analisis AI")
    st.markdown("---")
    st.markdown("**264.563** data titik harga")
    st.markdown("**35** provinsi · **30** komoditas")
    st.markdown("**2007–2023** · Model: Decision Tree")


# ─── FILTER DATA ────────────────────────────────────────────────────────────────
df_filtered = df.copy()
if sel_province != "NATIONAL AVERAGE":
    df_filtered = df_filtered[df_filtered["admin1"] == sel_province]
df_filtered = df_filtered[
    (df_filtered["year"] >= sel_year_range[0]) &
    (df_filtered["year"] <= sel_year_range[1])
]
if sel_category:
    df_filtered = df_filtered[df_filtered["category"].isin(sel_category)]


# ─── HEADER ─────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<p class="hero-title">🇮🇩 Dapur Data Indonesia</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Dashboard Harga Pangan Nasional · Powered by AI</p>', unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"📍 {sel_province.title()}")

# ─── KPI CARDS ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
avg_price = df_filtered["price"].mean()
max_price = df_filtered["price"].max()
mahal_pct = (df_filtered["label_harga"] == "Mahal").mean() * 100
top_comm = df_filtered.groupby("commodity")["price"].mean().idxmax()

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rata-rata Harga</div>
        <div class="metric-value">Rp {avg_price:,.0f}</div>
        <div class="metric-delta">Semua komoditas</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card" style="border-color:#DC2626;">
        <div class="metric-label">Harga Tertinggi</div>
        <div class="metric-value" style="color:#DC2626;">Rp {max_price:,.0f}</div>
        <div class="metric-delta">Dalam periode dipilih</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="metric-card" style="border-color:#D97706;">
        <div class="metric-label">Proporsi Label Mahal</div>
        <div class="metric-value" style="color:#D97706;">{mahal_pct:.1f}%</div>
        <div class="metric-delta">dari total transaksi</div>
    </div>""", unsafe_allow_html=True)
with k4:
    emoji = COMMODITY_EMOJI.get(top_comm, "📦")
    st.markdown(f"""
    <div class="metric-card" style="border-color:#7C3AED;">
        <div class="metric-label">Komoditas Termahal</div>
        <div class="metric-value" style="color:#7C3AED;font-size:1.1rem;">{emoji} {top_comm.title()[:20]}</div>
        <div class="metric-delta">Rata-rata tertinggi</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tren Harga",
    "🗺️ Peta Provinsi",
    "🤖 Prediksi AI",
    "🔥 Volatilitas & Insight",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1: TREN HARGA
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Perjalanan Harga dari Masa ke Masa</div>', unsafe_allow_html=True)

    col_sel, col_chart = st.columns([1, 3])
    with col_sel:
        sel_comm_trend = st.selectbox("Pilih Komoditas", COMMODITIES, key="trend_comm")
        compare_mode = st.checkbox("Bandingkan provinsi lain?")
        if compare_mode:
            prov_compare = st.multiselect("Provinsi", PROVINCES, default=PROVINCES[:3], max_selections=5)

    df_trend = df[df["commodity"] == sel_comm_trend].copy()

    with col_chart:
        if compare_mode and prov_compare:
            df_cmp = df_trend[df_trend["admin1"].isin(prov_compare)]
            df_cmp_agg = df_cmp.groupby(["year", "admin1"])["price"].mean().reset_index()
            fig_trend = px.line(
                df_cmp_agg, x="year", y="price", color="admin1",
                title=f"Tren Harga {sel_comm_trend.title()} per Provinsi",
                labels={"price": "Harga (Rp)", "year": "Tahun", "admin1": "Provinsi"},
                template="plotly_white",
            )
        else:
            df_nat = df_trend[df_trend["admin1"] == "NATIONAL AVERAGE"]
            df_nat_agg = df_nat.groupby("year")["price"].mean().reset_index()
            fig_trend = px.area(
                df_nat_agg, x="year", y="price",
                title=f"Tren Harga Nasional: {sel_comm_trend.title()}",
                labels={"price": "Harga (Rp)", "year": "Tahun"},
                template="plotly_white",
                color_discrete_sequence=["#2D6A4F"],
            )
            fig_trend.update_traces(fill="tozeroy", fillcolor="rgba(45,106,79,0.15)")

        # Mark COVID year
        fig_trend.add_vline(x=2020, line_dash="dash", line_color="#DC2626", opacity=0.5,
                            annotation_text="COVID-19", annotation_position="top right")
        fig_trend.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter"), height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Label distribution over years
    st.markdown("##### Distribusi Label Harga per Tahun")
    df_label_yr = df[df["commodity"] == sel_comm_trend].groupby(["year", "label_harga"]).size().reset_index(name="count")
    color_map = {"Murah": "#22C55E", "Sedang": "#F59E0B", "Mahal": "#EF4444"}
    fig_label = px.bar(
        df_label_yr, x="year", y="count", color="label_harga",
        barmode="stack", color_discrete_map=color_map,
        template="plotly_white", height=280,
        labels={"count": "Jumlah Pasar", "year": "Tahun", "label_harga": "Label"},
    )
    fig_label.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                            margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_label, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2: PETA PROVINSI
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Sebaran Harga Antar Provinsi</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        sel_comm_map = st.selectbox("Komoditas", COMMODITIES, key="map_comm")
        sel_yr_map = st.slider("Tahun", 2007, 2023, 2022, key="map_yr")
        metric_map = st.radio("Metrik", ["Rata-rata Harga", "% Label Mahal"], horizontal=True)

    df_map = df[(df["commodity"] == sel_comm_map) & (df["year"] == sel_yr_map)]
    df_map = df_map[df_map["admin1"] != "NATIONAL AVERAGE"]

    if metric_map == "Rata-rata Harga":
        df_agg = df_map.groupby("admin1")["price"].mean().reset_index()
        df_agg.columns = ["Provinsi", "Nilai"]
        color_col = "Nilai"
        color_scale = "YlGn"
        title_map = f"Rata-rata Harga {sel_comm_map.title()} ({sel_yr_map})"
    else:
        df_agg = df_map.groupby("admin1").apply(
            lambda x: (x["label_harga"] == "Mahal").mean() * 100
        ).reset_index()
        df_agg.columns = ["Provinsi", "Nilai"]
        color_col = "Nilai"
        color_scale = "RdYlGn_r"
        title_map = f"% Pasar Label MAHAL: {sel_comm_map.title()} ({sel_yr_map})"

    with c2:
        fig_bar = px.bar(
            df_agg.sort_values("Nilai", ascending=True).tail(20),
            x="Nilai", y="Provinsi", orientation="h",
            color="Nilai", color_continuous_scale=color_scale,
            title=title_map, template="plotly_white",
            labels={"Nilai": "Nilai", "Provinsi": ""},
        )
        fig_bar.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=500, margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap komoditas x provinsi
    st.markdown("##### 🗂️ Heatmap Harga: Semua Komoditas vs Provinsi")
    top_provs = df[df["admin1"] != "NATIONAL AVERAGE"].groupby("admin1")["price"].count().nlargest(15).index.tolist()
    df_heat = df[(df["admin1"].isin(top_provs)) & (df["year"] == sel_yr_map)]
    pivot = df_heat.groupby(["commodity", "admin1"])["price"].mean().unstack(fill_value=0)
    fig_heat = px.imshow(
        pivot, color_continuous_scale="YlGn", aspect="auto",
        labels=dict(color="Harga (Rp)"),
        title=f"Heatmap Harga Komoditas per Provinsi ({sel_yr_map})",
    )
    fig_heat.update_layout(
        height=420, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_tickangle=-35,
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3: PREDIKSI AI + CLAUDE ANALISIS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="ai-card"><h3>🤖 Mesin Tebak Harga</h3><p>Masukkan kombinasi provinsi, komoditas, dan waktu — model Decision Tree akan memprediksi label harga, lalu Claude AI akan memberikan analisis kontekstual.</p></div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        pred_province = st.selectbox("📍 Provinsi", PROVINCES, key="pred_prov")
        pred_commodity = st.selectbox("🛒 Komoditas", COMMODITIES, key="pred_comm")
        pred_year = st.slider("📅 Tahun", 2007, 2025, 2024, key="pred_yr")
        pred_month = st.selectbox("📆 Bulan", range(1, 13),
                                  format_func=lambda m: ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"][m-1],
                                  key="pred_month")
        run_pred = st.button("🔮 Prediksi Sekarang!", use_container_width=True, type="primary")

    with col_result:
        if run_pred:
            label, conf = predict_label(pred_year, pred_month, pred_province, pred_commodity)

            if label:
                badge_class = {"Mahal": "badge-mahal", "Sedang": "badge-sedang", "Murah": "badge-murah"}[label]
                emoji_map = {"Mahal": "🔴", "Sedang": "🟡", "Murah": "🟢"}

                st.markdown(f"""
                <div style="text-align:center; padding: 1.5rem; background:white; border-radius:16px; box-shadow:0 2px 12px rgba(0,0,0,0.08);">
                    <div style="font-size:3rem;">{emoji_map[label]}</div>
                    <div style="font-size:0.9rem; color:#6B7280; margin:0.5rem 0;">Prediksi Label Harga</div>
                    <span class="{badge_class}">{label.upper()}</span>
                    <div style="margin-top:1.2rem;">
                """, unsafe_allow_html=True)

                for lbl, prob in sorted(conf.items(), key=lambda x: -x[1]):
                    color = {"Mahal": "#EF4444", "Sedang": "#F59E0B", "Murah": "#22C55E"}[lbl]
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; margin:0.3rem 0; font-size:0.88rem;">
                        <span style="color:#374151;">{lbl}</span>
                        <span style="color:{color}; font-weight:600;">{prob*100:.1f}%</span>
                    </div>
                    <div style="height:8px; background:#F3F4F6; border-radius:99px; margin-bottom:0.3rem;">
                        <div style="width:{prob*100:.1f}%; height:100%; background:{color}; border-radius:99px;"></div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div></div>", unsafe_allow_html=True)

                # Claude AI Analysis
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 💬 Analisis Claude AI")

                # Get historical context
                hist = df[(df["admin1"] == pred_province) & (df["commodity"] == pred_commodity)]
                hist_summary = hist.groupby("year")["price"].mean().tail(5).to_dict()
                hist_str = ", ".join([f"{yr}: Rp{pr:,.0f}" for yr, pr in hist_summary.items()])

                prompt = f"""Kamu adalah analis ekonomi pangan Indonesia. Berikan analisis singkat dan tajam dalam Bahasa Indonesia (3-4 paragraf pendek).

Data:
- Provinsi: {pred_province}
- Komoditas: {pred_commodity}
- Periode: {["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"][pred_month-1]} {pred_year}
- Prediksi label harga (oleh model Decision Tree): {label} (confidence: {conf.get(label,0)*100:.1f}%)
- Rata-rata harga historis: {hist_str}

Tugas:
1. Jelaskan kenapa harga diprediksi {label} untuk komoditas ini di provinsi tersebut.
2. Sebutkan faktor musiman atau regional yang relevan.
3. Berikan 1 rekomendasi praktis untuk konsumen atau pedagang.

Gunakan gaya bahasa yang ringan tapi informatif. Jangan lebih dari 200 kata."""

                if api_key:
                    with st.spinner("Claude sedang menganalisis..."):
                        try:
                            analysis = ask_claude(prompt, api_key)
                            st.markdown(f'<div class="ai-response">{analysis}</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error API: {e}")
                else:
                    st.info("🔑 Masukkan Anthropic API Key di sidebar untuk mendapatkan analisis dari Claude AI.")
                    st.markdown(f'<div class="ai-response">**[Preview prompt yang akan dikirim ke Claude]**<br><br>{prompt}</div>', unsafe_allow_html=True)
            else:
                st.error("Kombinasi data tidak ditemukan di dataset. Coba pilihan lain.")
        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem; color:#9CA3AF;">
                <div style="font-size:3rem;">🔮</div>
                <p>Pilih provinsi, komoditas, dan periode<br>lalu klik <strong>Prediksi Sekarang!</strong></p>
            </div>
            """, unsafe_allow_html=True)

    # Batch prediction explorer
    st.markdown("---")
    st.markdown("##### 📊 Eksplorasi Prediksi Batch: 12 Bulan ke Depan")
    batch_prov = st.selectbox("Provinsi", PROVINCES, key="batch_prov")
    batch_comm = st.selectbox("Komoditas", COMMODITIES, key="batch_comm")
    batch_yr = st.slider("Tahun Prediksi", 2024, 2027, 2025, key="batch_yr")

    if st.button("🚀 Jalankan Prediksi 12 Bulan", use_container_width=True):
        results = []
        for m in range(1, 13):
            lbl, conf = predict_label(batch_yr, m, batch_prov, batch_comm)
            results.append({
                "Bulan": ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"][m-1],
                "Label": lbl,
                "P(Murah)": round(conf.get("Murah", 0)*100, 1),
                "P(Sedang)": round(conf.get("Sedang", 0)*100, 1),
                "P(Mahal)": round(conf.get("Mahal", 0)*100, 1),
            })
        df_batch = pd.DataFrame(results)
        color_map2 = {"Murah": "#22C55E", "Sedang": "#F59E0B", "Mahal": "#EF4444"}
        fig_batch = px.bar(
            df_batch.melt(id_vars=["Bulan","Label"], value_vars=["P(Murah)","P(Sedang)","P(Mahal)"],
                          var_name="Kelas", value_name="Probabilitas"),
            x="Bulan", y="Probabilitas", color="Kelas",
            color_discrete_map={"P(Murah)": "#22C55E", "P(Sedang)": "#F59E0B", "P(Mahal)": "#EF4444"},
            barmode="stack", template="plotly_white", height=300,
            title=f"Distribusi Prediksi {batch_comm.title()} di {batch_prov.title()} ({batch_yr})",
        )
        fig_batch.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_batch, use_container_width=True)
        st.dataframe(df_batch.set_index("Bulan"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4: VOLATILITAS & INSIGHT
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🔥 Komoditas Paling "Liar" & Insight Tersembunyi</div>', unsafe_allow_html=True)

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("##### Volatilitas Harga (Std. Deviasi per Komoditas)")
        vol = df.groupby("commodity")["price"].std().sort_values(ascending=False).reset_index()
        vol.columns = ["Komoditas", "Volatilitas"]
        vol["Emoji"] = vol["Komoditas"].apply(lambda x: COMMODITY_EMOJI.get(x, "📦"))
        fig_vol = px.bar(
            vol, x="Volatilitas", y="Komoditas", orientation="h",
            color="Volatilitas", color_continuous_scale="Reds",
            template="plotly_white", height=500,
        )
        fig_vol.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              coloraxis_showscale=False,
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_v2:
        st.markdown("##### Distribusi Label Harga per Kategori")
        df_cat_label = df.groupby(["category", "label_harga"]).size().reset_index(name="count")
        color_map3 = {"Murah": "#22C55E", "Sedang": "#F59E0B", "Mahal": "#EF4444"}
        fig_cat = px.bar(
            df_cat_label, x="count", y="category", color="label_harga",
            orientation="h", barmode="stack",
            color_discrete_map=color_map3, template="plotly_white", height=300,
            labels={"count": "Jumlah", "category": "", "label_harga": "Label"},
        )
        fig_cat.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("##### 📅 Bulan Paling Mahal (Rata-rata Semua Komoditas)")
        df_month = df.groupby("month")["price"].mean().reset_index()
        df_month["Bulan"] = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
        fig_month = px.bar(
            df_month, x="Bulan", y="price",
            color="price", color_continuous_scale="YlOrRd",
            template="plotly_white", height=250,
            labels={"price": "Rata-rata Harga (Rp)"},
        )
        fig_month.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                coloraxis_showscale=False,
                                margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_month, use_container_width=True)

    # Price spike detector
    st.markdown("---")
    st.markdown("##### 🚨 Deteksi Lonjakan Harga Ekstrem")
    threshold = st.slider("Threshold lonjakan (% kenaikan YoY)", 20, 100, 40)
    df_spike = df[df["admin1"] == "NATIONAL AVERAGE"].copy()
    df_spike = df_spike.groupby(["commodity", "year"])["price"].mean().reset_index()
    df_spike["pct_change"] = df_spike.groupby("commodity")["price"].pct_change() * 100
    spikes = df_spike[df_spike["pct_change"] > threshold].sort_values("pct_change", ascending=False)

    if not spikes.empty:
        fig_spike = px.scatter(
            spikes, x="year", y="commodity", size="pct_change", color="pct_change",
            color_continuous_scale="Reds", template="plotly_white",
            hover_data={"pct_change": ":.1f"},
            labels={"pct_change": "Kenaikan (%)", "year": "Tahun", "commodity": "Komoditas"},
            title=f"Lonjakan Harga > {threshold}% YoY (titik = besaran lonjakan)",
        )
        fig_spike.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380,
                                margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_spike, use_container_width=True)

        with st.expander(f"Lihat tabel ({len(spikes)} kejadian lonjakan)"):
            spikes_display = spikes.copy()
            spikes_display["pct_change"] = spikes_display["pct_change"].round(1).astype(str) + "%"
            spikes_display.columns = ["Komoditas", "Tahun", "Harga Rata-rata", "Kenaikan YoY"]
            st.dataframe(spikes_display, use_container_width=True)

    # Claude global insight
    st.markdown("---")
    st.markdown("##### 🧠 Minta Insight Global dari Claude AI")
    insight_topic = st.selectbox("Pilih topik analisis", [
        "Dampak COVID-19 terhadap harga pangan Indonesia",
        "Kenapa harga cabai sangat volatil di Indonesia?",
        "Kesenjangan harga pangan antar provinsi",
        "Tren harga beras 2007-2023 dan implikasinya",
        "Rekomendasi kebijakan stabilisasi harga pangan",
    ])
    if st.button("💡 Minta Analisis Claude", use_container_width=True):
        if api_key:
            context_prompt = f"""Kamu adalah ekonom senior yang ahli dalam kebijakan pangan Indonesia. 
Berikan analisis mendalam tentang: "{insight_topic}"

Gunakan konteks bahwa dataset ini mencakup 30 komoditas pangan, 35 provinsi Indonesia, periode 2007-2023.
Berikan analisis 3-4 paragraf dalam Bahasa Indonesia yang informatif, data-driven, dan actionable.
Sertakan angka atau statistik perkiraan jika relevan."""
            with st.spinner("Claude sedang berpikir..."):
                try:
                    insight = ask_claude(context_prompt, api_key)
                    st.markdown(f'<div class="ai-response">{insight}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("🔑 Masukkan Anthropic API Key di sidebar untuk fitur ini.")

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9CA3AF; font-size:0.82rem; padding:1rem 0;">
    🍚 <strong>Dapur Data Indonesia</strong> · Dashboard Harga Pangan Nasional<br>
    Data: WFP / BPS 2007–2023 · Model: Decision Tree Classifier · AI: Claude claude-sonnet-4-20250514<br>
    Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
