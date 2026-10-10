"""
Shopper Spectrum — Customer Segmentation & Product Recommendations
Streamlit App  ·  2022-2025 E-Commerce Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
from pathlib import Path

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── LOAD ASSETS ───────────────────────────────────────────────────────────────
ASSET_DIR = Path(__file__).parent / "assets"

@st.cache_resource
def load_bundle():
    with open(ASSET_DIR / "model_bundle.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_bundle()
scaler     = bundle["scaler"]
kmeans     = bundle["kmeans"]
label_map  = bundle["label_map"]
cos_sim_df = bundle["cos_sim_df"]
rfm        = bundle["rfm"]

seg_stats  = pd.read_csv(ASSET_DIR / "segment_stats.csv")

PRODUCT_LIST = sorted(cos_sim_df.index.tolist())

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1117;
    color: #E6EDF3;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #21262D;
}
section[data-testid="stSidebar"] * { color: #E6EDF3 !important; }

/* ── Metric Cards ── */
div[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label { color: #8B949E !important; font-size: 0.78rem; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #58A6FF !important; font-size: 1.6rem; font-weight: 700;
}

/* ── Cards ── */
.segment-card {
    background: #161B22;
    border-radius: 12px;
    padding: 20px 24px;
    border-left: 4px solid var(--seg-color, #58A6FF);
    margin-bottom: 12px;
}
.rec-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.rec-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: #58A6FF;
    min-width: 28px;
}
.rec-bar-bg {
    height: 6px; border-radius: 3px; background: #21262D; flex: 1;
}
.rec-bar-fill {
    height: 6px; border-radius: 3px; background: #58A6FF;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.tag-high   { background: #1c3a5e; color: #58A6FF; }
.tag-regular{ background: #1a3a2a; color: #3FB950; }
.tag-occ    { background: #3a2e0a; color: #D29922; }
.tag-risk   { background: #3a1a1a; color: #F78166; }

/* ── Section headers ── */
h1, h2, h3 { color: #E6EDF3 !important; }
.section-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #58A6FF;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.divider { border-top: 1px solid #21262D; margin: 28px 0; }

/* ── Input ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div { background: #161B22 !important; border-radius: 8px; }

/* ── Tabs ── */
button[data-baseweb="tab"] { background: transparent !important; color: #8B949E !important; font-size: 0.9rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: #E6EDF3 !important; border-bottom: 2px solid #58A6FF !important; }

/* ── Dataframe ── */
div[data-testid="stDataFrame"] { background: #161B22; border-radius: 10px; }

/* ── Button ── */
div.stButton > button {
    background: #1F6FEB;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    font-size: 0.9rem;
    transition: background 0.2s;
}
div.stButton > button:hover { background: #388BFD; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
SEG_META = {
    "High-Value": {
        "color": "#58A6FF", "icon": "🏆", "badge": "tag-high",
        "desc": "Recent, frequent, and high-spend customers — your VIPs.",
        "action": "Reward with exclusive loyalty perks and early-access offers.",
    },
    "Regular": {
        "color": "#3FB950", "icon": "📦", "badge": "tag-regular",
        "desc": "Steady purchasers with consistent spend — the reliable backbone.",
        "action": "Nurture with personalised cross-sell and moderate discounts.",
    },
    "Occasional": {
        "color": "#D29922", "icon": "🕐", "badge": "tag-occ",
        "desc": "Infrequent buyers who spend moderately when they do purchase.",
        "action": "Re-engage with targeted seasonal campaigns and bundles.",
    },
    "At-Risk": {
        "color": "#F78166", "icon": "⚠️", "badge": "tag-risk",
        "desc": "Customers who haven't bought recently — churn risk is high.",
        "action": "Send win-back emails with time-limited incentives.",
    },
}

def predict_segment(recency, frequency, monetary):
    import numpy as np
    row = np.array([[recency, np.log1p(frequency), np.log1p(monetary)]])
    X   = scaler.transform(row)
    cluster = kmeans.predict(X)[0]
    return label_map[cluster]

def get_recommendations(product_name: str, top_n: int = 5):
    if product_name not in cos_sim_df.index:
        matches = [p for p in cos_sim_df.index if product_name.upper() in p.upper()]
        if not matches:
            return []
        product_name = matches[0]
    sims = cos_sim_df[product_name].drop(product_name).sort_values(ascending=False)
    return [(prod, score) for prod, score in sims.head(top_n).items()]

def seg_badge(seg):
    m = SEG_META[seg]
    return f'<span class="badge {m["badge"]}">{m["icon"]} {seg}</span>'

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛍️ Shopper Spectrum use for shop.Fahad")
    st.markdown(
        '<p style="color:#8B949E;font-size:0.82rem;">E-Commerce Analytics Platform<br>'
        'Dataset: 2022–2023  ·  ~800 customers</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["📊  Overview", "🔍  Segmentation Tool", "🎯  Recommendations", "📈  Segment Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<p style="color:#8B949E;font-size:0.75rem;">'
        'KMeans · k=4 · RFM features<br>'
        'Item-based Cosine Similarity</p>',
        unsafe_allow_html=True,
    )

page = page.split("  ", 1)[1]   # strip icon prefix

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<p class="section-eyebrow">Shopper Spectrum..Fahad</p>', unsafe_allow_html=True)
    st.title("E-Commerce Customer Intelligence")
    st.markdown(
        "Unsupervised ML pipeline combining **RFM analysis**, **KMeans clustering**, "
        "and **item-based collaborative filtering** to understand and act on customer behaviour.",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # KPI row
    total_customers = rfm.shape[0]
    total_revenue   = rfm["Monetary"].sum()
    avg_frequency   = rfm["Frequency"].mean()
    avg_recency     = rfm["Recency"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",   f"{total_customers:,}")
    c2.metric("Total Revenue",     f"£{total_revenue:,.0f}")
    c3.metric("Avg Transactions",  f"{avg_frequency:.1f}")
    c4.metric("Avg Recency (days)",f"{avg_recency:.0f}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Segment summary cards
    st.markdown("### Segment Overview")
    cols = st.columns(4)
    for col, seg in zip(cols, ["High-Value","Regular","Occasional","At-Risk"]):
        meta  = SEG_META[seg]
        row   = seg_stats[seg_stats["Segment"] == seg].iloc[0]
        with col:
            st.markdown(
                f'<div class="segment-card" style="--seg-color:{meta["color"]}">'
                f'<div style="font-size:1.6rem">{meta["icon"]}</div>'
                f'<div style="font-size:1.1rem;font-weight:700;margin:4px 0">{seg}</div>'
                f'<div style="font-size:0.78rem;color:#8B949E;margin-bottom:10px">{meta["desc"]}</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:{meta["color"]}">{int(row.Count)}</div>'
                f'<div style="font-size:0.75rem;color:#8B949E">customers</div>'
                f'<div class="divider" style="margin:10px 0"></div>'
                f'<div style="font-size:0.78rem;color:#8B949E">Avg R: <b style="color:#E6EDF3">{row.Avg_R:.0f}d</b> &nbsp;'
                f'F: <b style="color:#E6EDF3">{row.Avg_F:.0f}</b> &nbsp;'
                f'M: <b style="color:#E6EDF3">£{row.Avg_M:,.0f}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # EDA plots
    st.markdown("### Exploratory Analysis")
    t1, t2, t3 = st.tabs(["Revenue Trends", "Customer Clusters", "RFM Distributions"])

    with t1:
        c_a, c_b = st.columns(2)
        with c_a:
            st.image(str(ASSET_DIR / "monthly_trend.png"), use_container_width=True)
        with c_b:
            st.image(str(ASSET_DIR / "country_revenue.png"), use_container_width=True)
        st.image(str(ASSET_DIR / "top_products.png"), use_container_width=True)

    with t2:
        c_a, c_b = st.columns(2)
        with c_a:
            st.image(str(ASSET_DIR / "cluster_scatter.png"), use_container_width=True)
        with c_b:
            st.image(str(ASSET_DIR / "segment_pie.png"), use_container_width=True)
        st.image(str(ASSET_DIR / "elbow_silhouette.png"), use_container_width=True)

    with t3:
        st.image(str(ASSET_DIR / "rfm_dist.png"), use_container_width=True)
        st.image(str(ASSET_DIR / "radar_chart.png"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SEGMENTATION TOOL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Segmentation Tool":
    st.markdown('<p class="section-eyebrow">Predict</p>', unsafe_allow_html=True)
    st.title("Customer Segmentation Tool")
    st.markdown(
        "Enter a customer's RFM values to predict their segment using the trained KMeans model.",
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_form, col_ref = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### Input RFM Values")
        recency   = st.number_input("Recency  (days since last purchase)", min_value=0,   max_value=730, value=30,   step=1)
        frequency = st.number_input("Frequency  (number of orders)",       min_value=1,   max_value=200, value=12,   step=1)
        monetary  = st.number_input("Monetary  (total spend, £)",          min_value=0.0, max_value=50000.0, value=800.0, step=10.0)

        predict_btn = st.button("Predict Segment", use_container_width=True)

        if predict_btn:
            seg  = predict_segment(recency, frequency, monetary)
            meta = SEG_META[seg]
            st.markdown(
                f'<div class="segment-card" style="--seg-color:{meta["color"]};margin-top:20px">'
                f'<div style="font-size:2.2rem">{meta["icon"]}</div>'
                f'<div style="font-size:1.5rem;font-weight:700;color:{meta["color"]};margin:6px 0">{seg}</div>'
                f'<div style="font-size:0.9rem;color:#C9D1D9;margin-bottom:12px">{meta["desc"]}</div>'
                f'<div style="font-size:0.8rem;background:#0D1117;padding:10px 14px;border-radius:8px;">'
                f'💡 <b>Recommended action:</b><br>{meta["action"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_ref:
        st.markdown("#### Segment Reference")
        for seg, meta in SEG_META.items():
            row = seg_stats[seg_stats["Segment"] == seg].iloc[0]
            st.markdown(
                f'<div class="segment-card" style="--seg-color:{meta["color"]}">'
                f'{meta["icon"]} <b>{seg}</b> &nbsp; '
                f'<span style="font-size:0.8rem;color:#8B949E">'
                f'R≈{row.Avg_R:.0f}d · F≈{row.Avg_F:.0f} orders · M≈£{row.Avg_M:,.0f}'
                f'</span><br>'
                f'<span style="font-size:0.8rem;color:#8B949E">{meta["desc"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.markdown('<p class="section-eyebrow">Discover</p>', unsafe_allow_html=True)
    st.title("Product Recommendation Engine")
    st.markdown(
        "Item-based collaborative filtering using **cosine similarity** on the "
        "customer–product purchase matrix.",
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_input, col_results = st.columns([1, 1.4], gap="large")

    with col_input:
        st.markdown("#### Select a Product")
        selected = st.selectbox(
            "Product",
            PRODUCT_LIST,
            index=PRODUCT_LIST.index("JUMBO BAG RED RETROSPOT") if "JUMBO BAG RED RETROSPOT" in PRODUCT_LIST else 0,
            label_visibility="collapsed",
        )
        top_n = st.slider("Number of recommendations", 3, 10, 5)
        rec_btn = st.button("Find Similar Products", use_container_width=True)

    with col_results:
        if rec_btn or True:   # show on load too
            recs = get_recommendations(selected, top_n)
            if recs:
                st.markdown(f"#### Recommendations for  `{selected}`")
                for i, (prod, score) in enumerate(recs, 1):
                    bar_w = int(score * 100)
                    st.markdown(
                        f'<div class="rec-card">'
                        f'  <div class="rec-rank">#{i}</div>'
                        f'  <div style="flex:1">'
                        f'    <div style="font-size:0.9rem;font-weight:600;margin-bottom:5px">{prod}</div>'
                        f'    <div class="rec-bar-bg">'
                        f'      <div class="rec-bar-fill" style="width:{bar_w}%"></div>'
                        f'    </div>'
                        f'  </div>'
                        f'  <div style="font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#8B949E;min-width:44px;text-align:right">'
                        f'    {score:.2f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No recommendations found for this product.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### How It Works")
    st.markdown(
        """
        1. **Purchase matrix** — a CustomerID × Product matrix is built, counting quantity purchased.
        2. **Item vectors** — each product becomes a vector of customer purchase frequencies.
        3. **Cosine similarity** — pairwise similarity is computed across all product pairs.
        4. **Lookup** — for a given product, the top-N most similar products are returned.
        """
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SEGMENT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Segment Explorer":
    st.markdown('<p class="section-eyebrow">Explore</p>', unsafe_allow_html=True)
    st.title("Segment Explorer")
    st.markdown("Browse and filter all customers by their predicted segment.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    seg_filter = st.multiselect(
        "Filter by Segment",
        ["High-Value","Regular","Occasional","At-Risk"],
        default=["High-Value","Regular","Occasional","At-Risk"],
    )

    filtered = rfm[rfm["Segment"].isin(seg_filter)].copy()
    filtered["Recency"]   = filtered["Recency"].astype(int)
    filtered["Frequency"] = filtered["Frequency"].astype(int)
    filtered["Monetary"]  = filtered["Monetary"].round(2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers shown",    f"{len(filtered):,}")
    c2.metric("Avg Monetary (£)",   f"£{filtered['Monetary'].mean():,.0f}")
    c3.metric("Avg Recency (days)", f"{filtered['Recency'].mean():.0f}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    display_df = filtered[["CustomerID","Recency","Frequency","Monetary","Segment"]].sort_values(
        "Monetary", ascending=False
    ).reset_index(drop=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=420,
        column_config={
            "CustomerID": st.column_config.TextColumn("Customer ID"),
            "Recency":    st.column_config.NumberColumn("Recency (days)", format="%d"),
            "Frequency":  st.column_config.NumberColumn("Orders",         format="%d"),
            "Monetary":   st.column_config.NumberColumn("Revenue (£)",    format="£%.2f"),
            "Segment":    st.column_config.TextColumn("Segment"),
        },
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Segment Statistics")
    shown_stats = seg_stats[seg_stats["Segment"].isin(seg_filter)]
    st.dataframe(shown_stats, use_container_width=True, hide_index=True)
