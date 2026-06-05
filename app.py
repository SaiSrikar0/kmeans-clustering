import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    kmeans     = joblib.load("kmeans_model.pkl")
    meta       = joblib.load("cluster_meta.pkl")
    X_train    = np.load("X_train.npy")
    lbl_train  = np.load("labels_train.npy")
    return kmeans, meta, X_train, lbl_train

kmeans, meta, X_train, lbl_train = load_model()
COLORS = meta["colors"]          # one per cluster
K      = meta["k"]
NEW_COLOR = "#FF00FF"            # magenta — new point stands out

# ── Cluster segment names ─────────────────────────────────────────────────────
SEGMENT_NAMES = {
    0: "💰 High Income, Low Spender",
    1: "🎯 High Income, High Spender",
    2: "📊 Mid Income, Mid Spender",
    3: "💸 Low Income, High Spender",
    4: "🏦 Low Income, Low Spender",
}

def label_cluster(kmeans, X_train, lbl_train, names):
    """Map model cluster IDs → segment names by centroid position."""
    centers = kmeans.cluster_centers_
    # Sort by income (x) then spending (y) heuristic
    ranked = sorted(range(K), key=lambda i: (centers[i][0], centers[i][1]))
    # Fixed labeling based on centroid quadrant
    mapping = {}
    for cid in range(K):
        cx, cy = centers[cid]
        if cx > 60 and cy < 45:
            mapping[cid] = 0   # High income, Low spend
        elif cx > 60 and cy >= 45:
            mapping[cid] = 1   # High income, High spend
        elif 40 <= cx <= 75 and 40 <= cy <= 65:
            mapping[cid] = 2   # Mid-mid
        elif cx < 50 and cy >= 55:
            mapping[cid] = 3   # Low income, High spend
        else:
            mapping[cid] = 4   # Low income, Low spend
    return mapping

seg_map = label_cluster(kmeans, X_train, lbl_train, SEGMENT_NAMES)

# ── Plotting function ─────────────────────────────────────────────────────────
def make_plot(new_income=None, new_spending=None, pred_cluster=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#1A1D24")

    # Existing clusters
    for i in range(K):
        mask = lbl_train == i
        ax.scatter(
            X_train[mask, 0], X_train[mask, 1],
            s=65, color=COLORS[i], alpha=0.75,
            edgecolors="white", linewidths=0.4,
            label=f"Cluster {i}"
        )

    # Centroids
    centers = kmeans.cluster_centers_
    ax.scatter(
        centers[:, 0], centers[:, 1],
        s=220, c="white", marker="X", zorder=5,
        edgecolors="black", linewidths=0.8, label="Centroids"
    )

    # New input point
    if new_income is not None:
        ax.scatter(
            new_income, new_spending,
            s=350, color=NEW_COLOR, marker="*", zorder=10,
            edgecolors="white", linewidths=1.2,
            label=f"New Customer → Cluster {pred_cluster}"
        )
        # Draw dashed line to centroid
        cx, cy = centers[pred_cluster]
        ax.plot(
            [new_income, cx], [new_spending, cy],
            "--", color=NEW_COLOR, linewidth=1.5, alpha=0.6
        )

    ax.set_xlabel("Annual Income (k$)", color="white", fontsize=12)
    ax.set_ylabel("Spending Score (1-100)", color="white", fontsize=12)
    ax.set_title("Mall Customer Segments — KMeans (K=5)", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, alpha=0.15, color="white")
    ax.legend(
        loc="upper left",
        facecolor="#1A1D24", edgecolor="#555",
        labelcolor="white", fontsize=9
    )
    plt.tight_layout()
    return fig

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#F0F0F0;'>🛍️ Mall Customer Segmentation</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#AAA; margin-top:-10px;'>KMeans Clustering — K=5 &nbsp;|&nbsp; Features: Annual Income & Spending Score</p>",
    unsafe_allow_html=True
)
st.markdown("---")

col_left, col_right = st.columns([1, 2.5], gap="large")

with col_left:
    st.markdown("### 🔍 Predict a New Customer")

    income   = st.slider("Annual Income (k$)",   min_value=0,   max_value=150, value=60, step=1)
    spending = st.slider("Spending Score (1-100)", min_value=0, max_value=100, value=50, step=1)

    predict_btn = st.button("🚀 Predict Cluster", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 🗺️ Segment Legend")
    for i in range(K):
        seg_idx = seg_map.get(i, i)
        seg_name = SEGMENT_NAMES.get(seg_idx, f"Cluster {i}")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
            f"<div style='width:14px;height:14px;border-radius:50%;background:{COLORS[i]};'></div>"
            f"<span style='color:#DDD;font-size:13px;'>Cluster {i} — {seg_name}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin:8px 0;'>"
        f"<div style='width:14px;height:14px;border-radius:50%;background:{NEW_COLOR};'></div>"
        f"<span style='color:#DDD;font-size:13px;'><b>★ New Customer</b></span>"
        f"</div>",
        unsafe_allow_html=True
    )

with col_right:
    if predict_btn:
        point = np.array([[income, spending]])
        pred  = int(kmeans.predict(point)[0])

        # Result card
        seg_idx  = seg_map.get(pred, pred)
        seg_name = SEGMENT_NAMES.get(seg_idx, f"Cluster {pred}")
        st.markdown(
            f"""
            <div style='
                background: linear-gradient(135deg, #1f1f2e, #2a2a40);
                border: 1.5px solid {COLORS[pred]};
                border-radius: 12px;
                padding: 18px 22px;
                margin-bottom: 18px;
            '>
                <h3 style='color:{COLORS[pred]}; margin:0 0 6px 0;'>
                    🎯 Predicted: Cluster {pred}
                </h3>
                <p style='color:#EEE; margin:0; font-size:15px;'>
                    {seg_name}
                </p>
                <p style='color:#AAA; margin:6px 0 0 0; font-size:13px;'>
                    Income: <b style='color:#FFF'>{income}k</b> &nbsp;|&nbsp;
                    Spending Score: <b style='color:#FFF'>{spending}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig = make_plot(income, spending, pred)
        st.pyplot(fig, use_container_width=True)
    else:
        # Default plot without new point
        fig = make_plot()
        st.pyplot(fig, use_container_width=True)
        st.info("👈 Adjust the sliders and hit **Predict Cluster** to see where your customer lands.")
