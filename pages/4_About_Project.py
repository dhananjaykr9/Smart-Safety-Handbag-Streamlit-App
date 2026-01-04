import streamlit as st
import base64, os

st.set_page_config(
    page_title="About Project – SafeBag",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Base64 Loader ----------
def load_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_base64("assets/logo2.png")

# ---------- Styling ----------
st.markdown("""
<style>
.about-title{font-size:38px;font-weight:800;
background:linear-gradient(90deg,#66b3ff,#4dffff);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
text-align:center;margin-bottom:8px;}
.about-sub{text-align:center;font-size:18px;color:#d0d3dc;margin-bottom:25px;}
.about-card{background:#2d303e;padding:22px;border-radius:16px;
border:1px solid #444859;box-shadow:0 8px 18px rgba(0,0,0,.35);
margin-bottom:22px;color:#f1f3f5;}
.flow-box{background:#1f2230;padding:14px;border-radius:12px;
text-align:center;font-weight:600;color:#7bed9f;margin-bottom:14px;}
.centered-img{display:block;margin:auto;border-radius:14px;}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
if logo_base64:
    st.markdown(f'<img src="data:image/png;base64,{logo_base64}" width="300" class="centered-img">', unsafe_allow_html=True)

st.markdown('<div class="about-title">SafeBag – Smart Safety Handbag</div>', unsafe_allow_html=True)
st.markdown('<div class="about-sub">IoT – Firebase – Machine Learning – GIS Integrated Women Safety Prototype</div>', unsafe_allow_html=True)
st.divider()

# ---------- Objective ----------
st.markdown("""
<div class="about-card" style="border-left:6px solid #ff6b6b;">
<h3>🎯 Project Objective</h3>
SafeBag is a prototype women-safety system that demonstrates how a wearable IoT device can
integrate with cloud infrastructure, machine learning models and geospatial routing
to provide situational crime-risk awareness and safer navigation support.
</div>
""", unsafe_allow_html=True)

# ---------- System Architecture ----------
st.markdown("""
<div class="about-card" style="border-left:6px solid #ffa502;">
<h3>🏗 System Architecture</h3>
<ul>
<li><b>ESP32 IoT Device</b> – GPS, PIR sensor and SOS button simulate emergency events.</li>
<li><b>Firebase Realtime Database</b> – Central cloud layer for event ingestion and state sharing.</li>
<li><b>ML Firebase Worker (Streamlit)</b> – Polls Firebase, performs offline Random Forest inference.</li>
<li><b>Routing Engine</b> – GraphHopper API + OpenStreetMap road network.</li>
<li><b>Dashboard</b> – Multi-page Streamlit application.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------- Workflow ----------
st.markdown("""
<div class="about-card" style="border-left:6px solid #2ecc71;">
<h3>🔁 Operational Workflow</h3>
<div class="flow-box">
IoT Event ➜ Firebase ➜ ML Firebase Worker ➜ Risk Prediction ➜ Safe Route Generation ➜ Web Dashboard
</div>
<ol>
<li>ESP32 or demo coordinates push location events into Firebase.</li>
<li>ML Worker page fetches unprocessed events and predicts crime type & risk level.</li>
<li>Results are written back to Firebase.</li>
<li>Live Routing page retrieves predictions and computes shortest & safest routes.</li>
<li>Users visualize ward-based demo routes or real GPS paths.</li>
</ol>
</div>
""", unsafe_allow_html=True)

# ---------- Data & ML ----------
st.markdown("""
<div class="about-card" style="border-left:6px solid #4dabf7;">
<h3>📊 Data & Machine Learning</h3>
<ul>
<li>Synthetic Nagpur crime dataset generated using NCRB-style distributions.</li>
<li>Feature engineering includes time-of-day, ward, latitude & longitude.</li>
<li>Random Forest classifiers trained for:
    <ul>
        <li>Crime Type Prediction</li>
        <li>Risk Level Estimation</li>
    </ul>
</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------- Prototype Disclaimer ----------
st.markdown("""
<div class="about-card" style="border-left:6px solid #e056fd;">
<h3>⚠ Prototype Disclaimer</h3>
This system is a <b>B.Tech Final Year Academic Prototype</b> developed for demonstration,
evaluation and research publication only. It is not intended for live law-enforcement
deployment or public safety use.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2026 SafeBag – Smart Safety Handbag | JD College of Engineering & Management, Nagpur")
