import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import pickle, warnings, os
warnings.filterwarnings('ignore')

os.makedirs("/home/claude/assets", exist_ok=True)

# ── PALETTE ──────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":        "#0D1117",
    "surface":   "#161B22",
    "accent1":   "#58A6FF",   # blue
    "accent2":   "#3FB950",   # green
    "accent3":   "#D29922",   # amber
    "accent4":   "#F78166",   # coral
    "text":      "#E6EDF3",
    "muted":     "#8B949E",
}
CLUSTER_COLORS = ["#58A6FF", "#3FB950", "#D29922", "#F78166"]
plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["muted"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "text.color":        PALETTE["text"],
    "grid.color":        "#21262D",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
df = pd.read_csv("/home/claude/ecommerce_data.csv")
print(f"  Raw shape: {df.shape}")

raw_shape = df.shape
df.dropna(subset=["CustomerID"], inplace=True)
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]
df["CustomerID"]  = df["CustomerID"].astype(str)
clean_shape = df.shape
print(f"  Clean shape: {clean_shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating EDA plots …")

# 2a  Top 10 countries by revenue
country_rev = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(country_rev.index[::-1], country_rev.values[::-1],
               color=PALETTE["accent1"], edgecolor="none", height=0.6)
for bar, val in zip(bars, country_rev.values[::-1]):
    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
            f"£{val:,.0f}", va="center", fontsize=8, color=PALETTE["muted"])
ax.set_title("Revenue by Country  (Top 10)", fontsize=13, pad=12, color=PALETTE["text"])
ax.set_xlabel("Total Revenue (£)")
ax.set_xlim(0, country_rev.max() * 1.18)
ax.grid(axis="x"); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("/home/claude/assets/country_revenue.png", dpi=130, bbox_inches="tight")
plt.close()

# 2b  Top 15 products
prod_rev = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(prod_rev.index[::-1], prod_rev.values[::-1],
               color=PALETTE["accent2"], edgecolor="none", height=0.7)
ax.set_title("Top 15 Products by Revenue", fontsize=13, pad=12)
ax.set_xlabel("Total Revenue (£)")
ax.set_xlim(0, prod_rev.max() * 1.15)
ax.grid(axis="x"); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("/home/claude/assets/top_products.png", dpi=130, bbox_inches="tight")
plt.close()

# 2c  Monthly revenue trend
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
monthly = df.groupby("YearMonth")["TotalPrice"].sum().reset_index()
monthly["YearMonth"] = monthly["YearMonth"].astype(str)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(monthly["YearMonth"], monthly["TotalPrice"],
        color=PALETTE["accent1"], linewidth=2)
ax.fill_between(range(len(monthly)), monthly["TotalPrice"],
                alpha=0.15, color=PALETTE["accent1"])
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly["YearMonth"], rotation=45, ha="right", fontsize=7)
ax.set_title("Monthly Revenue Trend  (2022–2023)", fontsize=13, pad=12)
ax.set_ylabel("Revenue (£)")
ax.grid(axis="y"); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("/home/claude/assets/monthly_trend.png", dpi=130, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. RFM FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing RFM …")
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency   = ("InvoiceDate",  lambda x: (snapshot_date - x.max()).days),
    Frequency = ("InvoiceNo",    "nunique"),
    Monetary  = ("TotalPrice",   "sum"),
).reset_index()

# 2d  RFM distributions
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
cols   = ["Recency", "Frequency", "Monetary"]
colors = [PALETTE["accent4"], PALETTE["accent2"], PALETTE["accent3"]]
for ax, col, color in zip(axes, cols, colors):
    ax.hist(rfm[col].clip(upper=rfm[col].quantile(0.99)),
            bins=40, color=color, edgecolor="none", alpha=0.85)
    ax.set_title(col, fontsize=12)
    ax.set_ylabel("Count")
    ax.grid(axis="y")
plt.suptitle("RFM Distributions", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("/home/claude/assets/rfm_dist.png", dpi=130, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
print("Clustering …")
scaler   = StandardScaler()
rfm_log  = rfm[["Recency","Frequency","Monetary"]].copy()
rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])
rfm_log["Monetary"]  = np.log1p(rfm_log["Monetary"])
X        = scaler.fit_transform(rfm_log)

# Elbow + silhouette
inertias   = []
sil_scores = []
K_range    = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(K_range), inertias, "o-", color=PALETTE["accent1"], linewidth=2)
axes[0].set_title("Elbow Curve", fontsize=13)
axes[0].set_xlabel("Number of Clusters (k)"); axes[0].set_ylabel("Inertia")
axes[0].axvline(4, color=PALETTE["accent4"], linestyle="--", linewidth=1.5, label="k=4")
axes[0].legend(); axes[0].grid()

axes[1].plot(list(K_range), sil_scores, "s-", color=PALETTE["accent2"], linewidth=2)
axes[1].set_title("Silhouette Score", fontsize=13)
axes[1].set_xlabel("Number of Clusters (k)"); axes[1].set_ylabel("Score")
axes[1].axvline(4, color=PALETTE["accent4"], linestyle="--", linewidth=1.5, label="k=4")
axes[1].legend(); axes[1].grid()
plt.tight_layout()
plt.savefig("/home/claude/assets/elbow_silhouette.png", dpi=130, bbox_inches="tight")
plt.close()

# Final model – k=4
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
km_final.fit(X)
rfm["Cluster"] = km_final.labels_

# Determine segment labels from cluster centroids
centers = pd.DataFrame(
    scaler.inverse_transform(
        np.column_stack([
            np.expm1(km_final.cluster_centers_[:,0]),
            np.expm1(km_final.cluster_centers_[:,1]),
            np.expm1(km_final.cluster_centers_[:,2]),
        ])
    ),
    columns=["Recency","Frequency","Monetary"]
)
# Higher Monetary + Frequency + lower Recency = High-Value
# Assign labels by ranking on a composite score
centers["score"] = -centers["Recency"] + centers["Frequency"]*5 + centers["Monetary"]/50
ranked = centers["score"].rank().astype(int)
label_map_raw = {
    centers["score"].idxmin():                                          "At-Risk",
    centers.loc[centers["Monetary"].idxmin()].name:                    "Occasional",
    centers.loc[centers.index.difference([centers["score"].idxmin(),
                                          centers["Monetary"].idxmin(),
                                          centers["score"].idxmax()])].index[0]: "Regular",
    centers["score"].idxmax():                                          "High-Value",
}
# Fallback: just rank by score
label_by_score  = {0:"At-Risk", 1:"Occasional", 2:"Regular", 3:"High-Value"}
score_order     = centers["score"].sort_values().index.tolist()
label_map_final = {idx: label_by_score[rank] for rank, idx in enumerate(score_order)}

rfm["Segment"] = rfm["Cluster"].map(label_map_final)
print(rfm["Segment"].value_counts())

# 2e  Cluster scatter (PCA)
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
fig, ax = plt.subplots(figsize=(8, 6))
for seg, color in zip(["High-Value","Regular","Occasional","At-Risk"], CLUSTER_COLORS):
    mask = rfm["Segment"] == seg
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, s=18,
               alpha=0.6, edgecolors="none", label=seg)
ax.set_title("Customer Segments  (PCA Projection)", fontsize=13, pad=12)
ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.legend(framealpha=0.2, edgecolor=PALETTE["muted"])
ax.grid(); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("/home/claude/assets/cluster_scatter.png", dpi=130, bbox_inches="tight")
plt.close()

# 2f  Radar / spider chart per segment
seg_means = rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean()
seg_norm  = (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min())
seg_norm["Recency"] = 1 - seg_norm["Recency"]   # invert: lower recency = better

labels    = ["Recency\n(Inverted)","Frequency","Monetary"]
angles    = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles   += angles[:1]
fig, axes = plt.subplots(1, 4, figsize=(14, 4), subplot_kw=dict(polar=True))
for ax, (seg, row), color in zip(axes, seg_norm.iterrows(),
                                  ["#58A6FF","#3FB950","#D29922","#F78166"]):
    vals = row.tolist() + row.tolist()[:1]
    ax.set_facecolor(PALETTE["surface"])
    ax.plot(angles, vals, color=color, linewidth=2)
    ax.fill(angles, vals, alpha=0.25, color=color)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_title(seg, fontsize=11, pad=14, color=color)
    ax.grid(color=PALETTE["muted"], alpha=0.4)
plt.suptitle("Segment RFM Profiles", fontsize=14, y=1.04)
plt.tight_layout()
plt.savefig("/home/claude/assets/radar_chart.png", dpi=130, bbox_inches="tight")
plt.close()

# 2g  Segment size pie
seg_counts = rfm["Segment"].value_counts()
fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    seg_counts.values,
    labels=seg_counts.index,
    autopct="%1.1f%%",
    colors=CLUSTER_COLORS,
    startangle=140,
    wedgeprops=dict(edgecolor=PALETTE["bg"], linewidth=2),
)
for t in texts:    t.set_color(PALETTE["text"])
for t in autotexts: t.set_color(PALETTE["bg"]); t.set_fontweight("bold")
ax.set_title("Segment Distribution", fontsize=13)
plt.tight_layout()
plt.savefig("/home/claude/assets/segment_pie.png", dpi=130, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. RECOMMENDATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
print("Building recommendation system …")
basket = df.groupby(["CustomerID","Description"])["Quantity"].sum().unstack(fill_value=0)
# Cosine similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
item_matrix  = basket.T          # products × customers
cos_sim      = cosine_similarity(item_matrix)
cos_sim_df   = pd.DataFrame(cos_sim, index=item_matrix.index, columns=item_matrix.index)

def get_recommendations(product_name: str, top_n: int = 5):
    if product_name not in cos_sim_df.index:
        # fuzzy fallback
        matches = [p for p in cos_sim_df.index if product_name.upper() in p.upper()]
        if not matches:
            return []
        product_name = matches[0]
    sims = cos_sim_df[product_name].drop(product_name).sort_values(ascending=False)
    return sims.head(top_n).reset_index().rename(
        columns={"Description":"Product","0":"Similarity",product_name:"Similarity"}
    ).to_dict("records")

# Test
print(get_recommendations("JUMBO BAG RED RETROSPOT"))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
print("Saving model artifacts …")
with open("/home/claude/assets/model_bundle.pkl","wb") as f:
    pickle.dump({
        "scaler":       scaler,
        "kmeans":       km_final,
        "label_map":    label_map_final,
        "cos_sim_df":   cos_sim_df,
        "rfm":          rfm,
        "cluster_stats": rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].agg(["mean","median"]).round(1),
    }, f)

# Also save small summary stats for the app
stats = rfm.groupby("Segment").agg(
    Count     = ("CustomerID","count"),
    Avg_R     = ("Recency",   "mean"),
    Avg_F     = ("Frequency", "mean"),
    Avg_M     = ("Monetary",  "mean"),
).round(1).reset_index()
stats.to_csv("/home/claude/assets/segment_stats.csv", index=False)
rfm.to_csv("/home/claude/assets/rfm_data.csv", index=False)

print("Pipeline complete ✓")
print(stats)
