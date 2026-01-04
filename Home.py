import streamlit as st
import base64, os, requests, time
from dotenv import load_dotenv

load_dotenv()
FIREBASE_URL = os.getenv("FIREBASE_URL")

st.set_page_config(
    page_title="Smart Safety Handbag",
    page_icon="👜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Base64 Loader ----------------
def load_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_base64("assets/logo2.png")

# ---------------- CSS ----------------
st.markdown("""
<style>
#splash-screen {
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at top, #102033, #0e1117 65%);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    animation: hideSplash 0.7s ease-out forwards;
    animation-delay: 2.5s;
}
.splash-content{text-align:center;}
.splash-logo{width:340px;max-width:85vw;animation:logoPop 1.4s ease-out forwards;}
.splash-title{font-size:2.7rem;font-weight:800;
background:linear-gradient(90deg,#66b3ff,#4dffff);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
opacity:0;animation:fadeUp 1s ease-out forwards;animation-delay:.9s;}
.splash-sub{font-size:1.2rem;color:#cfd8dc;opacity:0;
animation:fadeUp 1s ease-out forwards;animation-delay:1.3s;}
@keyframes logoPop{0%{opacity:0;transform:scale(.5)}60%{opacity:1;transform:scale(1.05)}100%{transform:scale(1)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes hideSplash{to{opacity:0;visibility:hidden}}

.title{font-size:2.9rem;font-weight:800;
background:linear-gradient(90deg,#66b3ff,#4dffff);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
text-align:center;}
.sub{text-align:center;font-size:1.25rem;color:#d0d3dc;}

.card{
    background:#2d303e;
    padding:22px;
    border-radius:16px;
    border:1px solid #444859;
    box-shadow:0 8px 18px rgba(0,0,0,.35);
    height:170px;                /* 🔒 FIXED HEIGHT */
    display:flex;
    flex-direction:column;
    justify-content:center;
}
.centered-img{display:block;margin:auto;border-radius:14px;}
</style>
""", unsafe_allow_html=True)

# ---------------- Splash Screen ----------------
if logo_base64:
    st.markdown(f"""
    <div id="splash-screen">
        <div class="splash-content">
            <img src="data:image/png;base64,{logo_base64}" class="splash-logo">
            <div class="splash-title">SafeBag</div>
            <div class="splash-sub">Smart Safety Handbag System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Header ----------------
if logo_base64:
    st.markdown(f'<img src="data:image/png;base64,{logo_base64}" width="220" class="centered-img">', unsafe_allow_html=True)

st.markdown('<div class="title">SafeBag: Smart Safety Handbag</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Real-Time IoT | Machine Learning Risk Prediction | GIS Safe Routing</div>', unsafe_allow_html=True)
st.divider()

# ---------------- System Status ----------------
st.subheader("🖥 System Status")

def check_firebase():
    try:
        r = requests.get(f"{FIREBASE_URL}/.json", timeout=5)
        return r.status_code == 200
    except:
        return False

def check_worker_alive():
    try:
        r = requests.get(f"{FIREBASE_URL}/system_status/ml_worker_heartbeat.json", timeout=5)
        if r.status_code != 200 or not r.json():
            return False
        ts = int(r.json().get("ts", 0))
        now = int(time.time() * 1000)
        return (now - ts) < 15000
    except:
        return False

fb_ok = check_firebase()
worker_ok = check_worker_alive()

s1, s2 = st.columns(2)
with s1: st.metric("Firebase Connection", "Online" if fb_ok else "Offline")
with s2: st.metric("ML Worker Status", "Running" if worker_ok else "Stopped")

st.divider()

# ---------------- Feature Cards ----------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="card" style="border-left:6px solid #ff6b6b;"><h4>📡 Safety Device</h4>ESP32-based smart handbag continuously transmits GPS, SOS and motion alerts to Firebase.</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card" style="border-left:6px solid #ffa502;"><h4>🤖 ML Risk Engine</h4>Crime-type classification and real-time risk scoring using Random Forest models.</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card" style="border-left:6px solid #2ecc71;"><h4>🗺 Safe Routing</h4>Crime-weighted routing using GraphHopper & OpenStreetMap to avoid hotspots.</div>', unsafe_allow_html=True)

st.divider()

# ---------------- Modules ----------------
st.subheader("📂 Dashboard Modules")
st.markdown("""
| Module | Function |
|------|----------|
| 🗺 **Live Routing** | Visualizes live GPS, hotspot zones & computes safest route |
| 📊 **Event Dashboard** | Displays raw ESP32 events stored in Firebase |
| 🤖 **ML Firebase Worker** | Executes background ML inference & updates Firebase |
| ℹ **About Project** | Architecture, methodology & academic overview |
""")

st.info("👈 Use the sidebar to open different modules.")
st.markdown("""<hr><center>© 2026 Smart Safety Handbag | JD College of Engineering & Management, Nagpur</center>""", unsafe_allow_html=True)
