"""
Real Estate Image Classifier — Streamlit Frontend
Connects to FastAPI at http://localhost:8080
Run with: streamlit run app.py
"""

import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8080"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EstateVision — AI Classifier",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background-color: #0b0c0e;
    color: #f0ead8;
}
section[data-testid="stSidebar"] {
    background-color: #111316 !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    font-size: 2.8rem !important;
    color: #f0ead8 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
}
h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 400 !important;
    color: #f0ead8 !important;
}
[data-testid="stFileUploader"] {
    background: #181a1e;
    border: 1.5px dashed rgba(201,169,110,0.35);
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(201,169,110,0.6);
}
.stButton > button {
    background: linear-gradient(135deg, #c9a96e, #a07840) !important;
    color: #0b0c0e !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(201,169,110,0.25) !important;
}
[data-testid="stMetric"] {
    background: #1e2025;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] {
    color: #7a7d85 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    color: #c9a96e !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2rem !important;
}
[data-testid="stProgress"] > div > div {
    background-color: #7a7d85 !important;
}
[data-testid="stProgress"] > div {
    background: transparent !important;
}
[data-testid="stProgress"] > div > div {
    background-color: #111316 !important;
    border-radius: 999px !important;
}
[data-testid="stProgress"] > div > div > div {
    background-color: #111316 !important;
    border-radius: 999px !important;
}
.stSuccess {
    background: rgba(76,175,125,0.1) !important;
    border: 1px solid rgba(76,175,125,0.25) !important;
    color: #4caf7d !important;
    border-radius: 8px !important;
}
.stError {
    background: rgba(224,82,82,0.1) !important;
    border: 1px solid rgba(224,82,82,0.25) !important;
    border-radius: 8px !important;
}
.stInfo {
    background: rgba(201,169,110,0.08) !important;
    border: 1px solid rgba(201,169,110,0.2) !important;
    color: #c9a96e !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] {
    background: #1e2025 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 1.25rem 0 !important;
}
.stCaption, small {
    color: #7a7d85 !important;
    font-size: 12px !important;
}
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}
section[data-testid="stSidebar"] .stCaption {
    color: #7a7d85 !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(201,169,110,0.25); border-radius: 4px; }
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #c9a96e, #a07840) !important;
}
.stProgress > div > div {
    background-color: #1e2025 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
CLASS_META = {
    "Bedroom":      "🛏️",
    "Coast":        "🌊",
    "Forest":       "🌲",
    "Highway":      "🛣️",
    "Industrial":   "🏭",
    "Inside city":  "🏙️",
    "Kitchen":      "🍳",
    "Living room":  "🛋️",
    "Mountain":     "⛰️",
    "Office":       "💼",
    "Open country": "🌾",
    "Store":        "🏪",
    "Street":       "🚶",
    "Suburb":       "🏡",
    "Tall building":"🏢",
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div style="padding: 0 0 1rem; border-bottom: 1px solid rgba(255,255,255,0.07);">
        <p style="font-family:'Cormorant Garamond',serif; font-size:1.6rem;
                  font-weight:300; color:#f0ead8; margin:0; letter-spacing:-0.01em;">
            Estate<span style="color:#c9a96e;">Vision</span>
        </p>
        <p style="font-size:10px; letter-spacing:0.2em; text-transform:uppercase;
                  color:#7a7d85; margin:2px 0 0;">AI Property Classifier</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:10px; letter-spacing:0.18em; text-transform:uppercase;
              color:#7a7d85; margin-bottom:0.6rem;"> &nbsp;Information</p>
    """, unsafe_allow_html=True)

    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        api_ok = True
    except Exception:
        health = {}
        api_ok = False

    if api_ok:
        st.markdown("""
        <div style="background:rgba(76,175,125,0.12); border:1px solid rgba(76,175,125,0.3);
                    border-radius:8px; padding:10px 14px; margin-bottom:0.75rem;
                    display:flex; align-items:center; gap:8px;">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                         background:#4caf7d;box-shadow:0 0 6px #4caf7d;flex-shrink:0;"></span>
            <span style="font-size:13px; color:#4caf7d; font-weight:400;">API connected</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background:#1e2025;border:1px solid rgba(255,255,255,0.07);
                        border-radius:8px;padding:10px 12px;text-align:center;margin-bottom:8px;">
                <p style="font-size:10px;color:#7a7d85;text-transform:uppercase;
                           letter-spacing:0.12em;margin:0 0 4px;">Model</p>
                <p style="font-size:13px;color:#c9a96e;font-weight:500;
                           margin:0;font-family:monospace;">{health.get('model','—')}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background:#1e2025;border:1px solid rgba(255,255,255,0.07);
                        border-radius:8px;padding:10px 12px;text-align:center;margin-bottom:8px;">
                <p style="font-size:10px;color:#7a7d85;text-transform:uppercase;
                           letter-spacing:0.12em;margin:0 0 4px;">Device</p>
                <p style="font-size:13px;color:#c9a96e;font-weight:500;
                           margin:0;font-family:monospace;">{health.get('device','—')}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#1e2025;border:1px solid rgba(255,255,255,0.07);
                    border-radius:8px;padding:10px 12px;
                    display:flex;align-items:center;justify-content:space-between;">
            <span style="font-size:12px;color:#7a7d85;text-transform:uppercase;
                          letter-spacing:0.1em;">Classes</span>
            <span style="font-size:13px;color:#c9a96e;font-weight:500;
                          font-family:monospace;">{health.get('num_classes','15')}</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:rgba(224,82,82,0.1); border:1px solid rgba(224,82,82,0.25);
                    border-radius:8px; padding:10px 14px; margin-bottom:0.75rem;">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                         background:#e05252;margin-right:8px;"></span>
            <span style="font-size:13px; color:#e05252;">API not available</span>
            <p style="font-size:11px;color:#7a7d85;margin:6px 0 0;line-height:1.5;">
            Start with:<br>
            <code style="color:#c9a96e;font-size:10px;">uvicorn main:app --reload --port 8080</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:1.25rem 0;border-top:1px solid rgba(255,255,255,0.07);"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:10px; letter-spacing:0.18em; text-transform:uppercase;
              color:#7a7d85; margin-bottom:0.75rem;"> &nbsp;Model categories</p>
    """, unsafe_allow_html=True)

    active_class = st.session_state.get("predicted_class", None)

    for name, emoji in CLASS_META.items():
        is_active = (name == active_class)
        bg     = "rgba(201,169,110,0.12)" if is_active else "#1e2025"
        border = "rgba(201,169,110,0.35)" if is_active else "rgba(255,255,255,0.07)"
        color  = "#c9a96e"                if is_active else "#7a7d85"
        weight = "500"                    if is_active else "300"
        dot    = f"""<span style="width:6px;height:6px;border-radius:50%;
                         background:#c9a96e;flex-shrink:0;display:inline-block;
                         margin-right:2px;"></span>""" if is_active else ""

        st.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:6px;
                    padding:7px 10px;margin-bottom:5px;
                    display:flex;align-items:center;gap:8px;">
            <span style="font-size:14px;">{emoji}</span>
            <span style="font-size:12px;color:{color};font-weight:{weight};flex:1;">{name}</span>
            {dot}
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding:2.5rem 0 0.5rem;">
    <p style="font-size:11px;letter-spacing:0.22em;text-transform:uppercase;
              color:#c9a96e;margin-bottom:0.5rem;">Real Estate Image Classification</p>
    <h1>Visual Recognition <em style="font-style:italic;color:#c9a96e;">Architectural</em></h1>
    <p style="color:#7a7d85;font-size:15px;font-weight:300;max-width:520px;
              line-height:1.7;margin-top:0.5rem;">
        Upload an image of a property and our model will classify it
        automatically into one of the 15 real estate categories.
    </p>
</div>
<div style="border-top:1px solid rgba(255,255,255,0.07);margin:1.5rem 0 2rem;"></div>
""", unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("""
    <p style="font-size:11px;letter-spacing:0.15em;text-transform:uppercase;
              color:#7a7d85;margin-bottom:0.5rem;">📷 Upload Image</p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag and drop or select an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if st.button("✦  Classify image", use_container_width=True):
            with st.spinner("Analyzing…"):
                try:
                    uploaded.seek(0)
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded.name, uploaded, uploaded.type)},
                        timeout=30,
                    )
                    if response.status_code == 200:
                        st.session_state["result"]          = response.json()
                        st.session_state["predicted_class"] = response.json()["predicted_class"]
                        st.rerun()
                    elif response.status_code == 400:
                        st.error(f"Image error: {response.json()['detail']}")
                    else:
                        st.error(f"Server error ({response.status_code})")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API.")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
    else:
        st.info("Upload an image to get started")

with col_result:
    st.markdown("""
    <p style="font-size:11px;letter-spacing:0.15em;text-transform:uppercase;
              color:#7a7d85;margin-bottom:0.5rem;">✦ Result</p>
    """, unsafe_allow_html=True)

    result = st.session_state.get("result", None)

    if result:
        cls   = result["predicted_class"]
        emoji = CLASS_META.get(cls, "📍")
        conf  = result["confidence"]

        st.markdown(f"""
        <div style="background:#1e2025;border:1px solid rgba(201,169,110,0.2);
                    border-radius:12px;padding:1.5rem 1.75rem;text-align:center;
                    margin-bottom:1rem;">
            <p style="font-size:11px;letter-spacing:0.18em;text-transform:uppercase;
                       color:#7a7d85;margin:0 0 0.4rem;">Detected category</p>
            <p style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;
                       font-weight:300;color:#f0ead8;margin:0 0 0.75rem;
                       letter-spacing:-0.01em;">{emoji} {cls}</p>
            <p style="font-family:'Cormorant Garamond',serif;font-size:2.6rem;
                       font-weight:300;color:#c9a96e;margin:0;line-height:1;">
                {conf*100:.1f}<span style="font-size:1.2rem;color:#8a6e42;">%</span>
            </p>
            <p style="font-size:10px;letter-spacing:0.15em;text-transform:uppercase;
                       color:#7a7d85;margin:4px 0 0;">Confidence</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size:10px;letter-spacing:0.15em;text-transform:uppercase;
                  color:#7a7d85;margin-bottom:0.6rem;">Top probabilities</p>
        """, unsafe_allow_html=True)

        sorted_probs = sorted(
            result["all_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        for i, (name, prob) in enumerate(sorted_probs):
            emoji_c = CLASS_META.get(name, "📍")
            pct     = prob * 100
            color   = "#c9a96e" if i == 0 else "#7a7d85"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <span style="font-size:13px;width:18px;">{emoji_c}</span>
                <span style="font-size:12px;color:{color};width:100px;
                              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                              font-weight:{'500' if i==0 else '300'};">{name}</span>
                <div style="flex:1;height:4px;background:#1e2025;border-radius:2px;overflow:hidden;">
                    <div style="width:{pct:.1f}%;height:100%;border-radius:2px;
                                background:{'#c9a96e' if i==0 else '#3a3d45'};
                                transition:width 0.8s;"></div>
                </div>
                <span style="font-size:12px;color:{color};width:40px;
                              text-align:right;font-weight:{'500' if i==0 else '300'};">
                    {pct:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("View all categories"):
            all_sorted = sorted(
                result["all_probabilities"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for name, prob in all_sorted:
                emoji_c = CLASS_META.get(name, "📍")
                pct     = prob * 100
                st.progress(prob, text=f"{emoji_c} {name}:  {pct:.2f}%")

    else:
        st.markdown("""
        <div style="background:#111316;border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;padding:3rem 2rem;text-align:center;">
            <p style="font-size:2rem;margin-bottom:0.5rem;opacity:0.3;">✦</p>
            <p style="font-family:'Cormorant Garamond',serif;font-size:1.3rem;
                       font-weight:300;color:#7a7d85;margin:0;">
                The result will appear here
            </p>
            <p style="font-size:12px;color:#4a4d55;margin-top:0.5rem;font-weight:300;">
                Upload an image and click classify
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.07);
            margin-top:3rem;padding:1.5rem 0;
            display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;gap:0.5rem;">
    <span style="font-size:12px;color:#4a4d55;font-weight:300;">
        <strong style="color:#7a7d85;font-weight:400;">EstateVision</strong>
        &nbsp;·&nbsp; &nbsp;·&nbsp; ICAI
    </span>
    <span style="font-size:11px;color:#4a4d55;letter-spacing:0.05em;">
        ResNeXt101-32x8d &nbsp;·&nbsp; Transfer Learning &nbsp;·&nbsp; PyTorch
    </span>
</div>
""", unsafe_allow_html=True)