from dotenv import load_dotenv
load_dotenv()

import os
import math
import datetime
import joblib
import requests
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
import networkx as nx
import osmnx as ox
import numpy as np
import json

# ---------- Config ----------
DATA_CSV = "data/nagpur_crime_data_synthetic.csv"
CRIME_MODEL_PKL = "models/crime_type_model.pkl"
RISK_MODEL_PKL = "models/risk_model.pkl"
GRAPH_FILE = "data/nagpur_graph.graphml"   # projected graph (meters) with optional 'safety_weight'
PLACE = "Nagpur, India"
SAVED_MAP_HTML = "data/last_map.html"

# GraphHopper API key
DEFAULT_GH_API_KEY = os.getenv("GH_API_KEY")

# Firebase Realtime DB base (no trailing slash).
FIREBASE_URL = os.environ.get("FIREBASE_URL", "https://smart-safety-handbag-default-rtdb.asia-southeast1.firebasedatabase.app")
FIREBASE_AUTH = os.environ.get("FIREBASE_AUTH", None)

# Device id to look for (same as ESP32)
DEVICE_ID = "handbag_001"

# ---------- Styling ----------
st.set_page_config(layout="wide", page_title="Smart Safety Handbag", page_icon="🛡️")

# Optimized CSS: Lightened backgrounds and increased contrast for Dark Mode
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* 1. Whole Page Background - Softened dark blue-gray */
    .stApp {
        background-color: #1a1c23;
        color: #e1e1e6;
        font-family: 'Inter', sans-serif;
    }

    /* 2. Control Panel (Sidebar) styling */
    [data-testid="stSidebar"] {
        background-color: #252832 !important;
        border-right: 1px solid #3d414d;
    }
    [data-testid="stSidebar"] * {
        color: #f0f0f5 !important;
    }
    
    /* Input field text color fix */
    input, [data-baseweb="select"] * {
        color: #ffffff !important; 
    }
    
    /* Input backgrounds lightened for better focus */
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: #343746 !important;
        border-color: #4b4f63 !important;
    }

    .app-title { 
        font-size: 34px; 
        font-weight: 800; 
        background: linear-gradient(90deg, #66b3ff, #4dffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px; 
    }
    
    /* Transparent metrics with bright text */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.08);
        padding: 15px;
        border-radius: 12px;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.15);
    }

    /* Cards styled with surface-level colors (lighter than background) */
    .card { 
        background: #2d303e; 
        padding: 20px; 
        border-radius: 15px; 
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #444859;
        color: #ffffff;
    }
    
    .card-risk { border-left: 6px solid #ff6b6b; }
    .card-crime { border-left: 6px solid #4dabf7; }
    
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #f1f3f5;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Ensure expanders and dataframes are readable */
    .stDataFrame, .stExpander, div[data-testid="stStatusWidget"] {
        background-color: #2d303e !important;
        border: 1px solid #444859 !important;
    }
    
    /* Map container wrapper to provide contrast against dark page */
    .map-wrapper {
        border: 2px solid #444859;
        border-radius: 15px;
        overflow: hidden;
        background-color: #ffffff; /* Frame background */
    }
</style>
""", unsafe_allow_html=True)


# ---------- Helpers (data/models/graph) ----------
@st.cache_data(ttl=60*60)
def load_data():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"CSV not found: {DATA_CSV}")
    return pd.read_csv(DATA_CSV)

@st.cache_resource
def load_artifact(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource
def ensure_graph(place=PLACE, graph_file=GRAPH_FILE):
    if os.path.exists(graph_file):
        Gp = ox.load_graphml(graph_file)
        try:
            G_latlon = ox.project_graph(Gp, to_crs="EPSG:4326")
        except Exception:
            G_latlon = ox.graph_from_place(place, network_type="drive")
        return Gp, G_latlon
    G_unproj = ox.graph_from_place(place, network_type="drive")
    Gp = ox.project_graph(G_unproj)
    ox.save_graphml(Gp, graph_file)
    G_latlon = ox.project_graph(Gp, to_crs="EPSG:4326")
    return Gp, G_latlon

def get_timeslot(hour):
    if 0 <= hour <= 5:
        return "Night"
    if 6 <= hour <= 11:
        return "Morning"
    if 12 <= hour <= 17:
        return "Afternoon"
    return "Evening"

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def prepare_features_for_artifact(lat, lon, timestamp, artifact):
    if isinstance(timestamp, str):
        timestamp = datetime.datetime.fromisoformat(timestamp)
    if timestamp is None:
        timestamp = datetime.datetime.now()
    hour = timestamp.hour
    day = timestamp.strftime("%A")
    slot = get_timeslot(hour)

    le_day = artifact.get("le_day") if artifact else None
    le_slot = artifact.get("le_slot") if artifact else None
    le_ward = artifact.get("le_ward") if artifact else None

    try:
        day_enc = int(le_day.transform([day])[0]) if le_day is not None else 0
    except Exception:
        day_enc = 0
    try:
        slot_enc = int(le_slot.transform([slot])[0]) if le_slot is not None else 0
    except Exception:
        slot_enc = 0

    ward_enc = 0
    features = artifact.get("feature_columns", ["Ward_enc","Latitude","Longitude","Hour","DayOfWeek_enc","TimeSlot_enc"]) if artifact else ["Ward_enc","Latitude","Longitude","Hour","DayOfWeek_enc","TimeSlot_enc"]
    row = []
    for f in features:
        if f == "Ward_enc":
            row.append(ward_enc)
        elif f == "Latitude":
            row.append(float(lat))
        elif f == "Longitude":
            row.append(float(lon))
        elif f == "Hour":
            row.append(int(hour))
        elif f == "DayOfWeek_enc":
            row.append(int(day_enc))
        elif f == "TimeSlot_enc":
            row.append(int(slot_enc))
        else:
            row.append(0)
    return pd.DataFrame([row], columns=features)

def safe_predict_proba(model, X):
    try:
        proba = model.predict_proba(X)[0]
        return np.asarray(proba, dtype=float)
    except Exception:
        try:
            _ = model.predict(X)
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                proba = np.zeros(len(classes), dtype=float)
                proba[0] = 1.0
                return proba
            return np.asarray([1.0])
        except Exception:
            return np.asarray([1.0])

def predict_with_artifact(lat, lon, timestamp, artifact):
    X = prepare_features_for_artifact(lat, lon, timestamp, artifact)
    if artifact is None or "model" not in artifact:
        raise RuntimeError("Artifact missing 'model' key or artifact is None.")
    model = artifact["model"]
    pred = model.predict(X)
    proba = safe_predict_proba(model, X)

    le = None
    for k in ("le_target","le_risk","le","le_label","le_y"):
        if artifact and k in artifact:
            le = artifact[k]; break

    proba_dict = {}
    label = pred[0]
    if le is not None and hasattr(le, "classes_"):
        classes = list(le.classes_)
        n = min(len(classes), len(proba))
        for i in range(n):
            proba_dict[classes[i]] = float(proba[i])
        for c in classes[n:]:
            proba_dict[c] = 0.0
        try:
            label = le.inverse_transform(pred)[0]
        except Exception:
            label = pred[0]
    else:
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            n = min(len(classes), len(proba))
            for i in range(n):
                proba_dict[classes[i]] = float(proba[i])
            for c in classes[n:]:
                proba_dict[c] = 0.0
        else:
            for i in range(len(proba)):
                proba_dict[str(i)] = float(proba[i])
    return str(label), proba_dict

def nearest_node_wrapper(G, lon, lat):
    try:
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)
    except TypeError:
        try:
            return ox.nearest_nodes(G, lon, lat)
        except Exception:
            best = None; best_d = float("inf")
            for n, data in G.nodes(data=True):
                if "x" in data and "y" in data:
                    d = haversine_km(lat, lon, data["y"], data["x"])
                    if d < best_d:
                        best_d = d; best = n
            if best is None:
                raise RuntimeError("Could not find nearest node (graph nodes lack coords).")
            return best

def route_coords_from_graph(G_latlon, route):
    coords = []
    if not route or len(route) < 2:
        return coords
    for u, v in zip(route[:-1], route[1:]):
        data = G_latlon.get_edge_data(u, v)
        if data is None:
            data = G_latlon.get_edge_data(v, u)
        seg = None
        if data is None:
            node_u = G_latlon.nodes[u]; node_v = G_latlon.nodes[v]
            seg = [(node_u.get("y"), node_u.get("x")), (node_v.get("y"), node_v.get("x"))]
        else:
            edge_data = None
            if isinstance(data, dict) and any(isinstance(k, (int, str)) for k in data.keys()):
                chosen = None
                for k, val in data.items():
                    if isinstance(val, dict) and "geometry" in val and val["geometry"] is not None:
                        chosen = val; break
                if chosen is None:
                    chosen = next(iter(data.values()))
                edge_data = chosen
            else:
                edge_data = data
            if edge_data is not None and isinstance(edge_data, dict) and "geometry" in edge_data and edge_data["geometry"] is not None:
                try:
                    seg = [(y, x) for x, y in edge_data["geometry"].coords]
                except Exception:
                    seg = None
            else:
                node_u = G_latlon.nodes[u]; node_v = G_latlon.nodes[v]
                seg = [(node_u.get("y"), node_u.get("x")), (node_v.get("y"), node_v.get("x"))]
        if seg:
            if coords and coords[-1] == seg[0]:
                coords.extend(seg[1:])
            else:
                coords.extend(seg)
    return coords

def get_route_graphhopper(start_lat, start_lon, end_lat, end_lon, api_key, profile="car"):
    if not api_key:
        raise RuntimeError("GraphHopper API key required.")
    url = "https://graphhopper.com/api/1/route"
    params = [
        ("point", f"{start_lat},{start_lon}"),
        ("point", f"{end_lat},{end_lon}"),
        ("vehicle", profile),
        ("points_encoded", "false"),
        ("key", api_key)
    ]
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if "paths" in data and len(data["paths"]) > 0:
        path0 = data["paths"][0]
        if "points" in path0 and "coordinates" in path0["points"]:
            coords = [(lat, lon) for lon, lat in path0["points"]["coordinates"]]
            return coords
    raise RuntimeError("GraphHopper returned no route.")

# ---------- Firebase helpers (robust, quiet) ----------
def _build_firebase_url(path):
    base = FIREBASE_URL.rstrip("/")
    url = f"{base}/{path}.json"
    if FIREBASE_AUTH:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}auth={FIREBASE_AUTH}"
    return url

def fetch_latest_event_node(device_id):
    try:
        url = _build_firebase_url(f"latest_events/{device_id}")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def fetch_recent_events(limit=50):
    url = FIREBASE_URL.rstrip("/") + "/events.json"
    try:
        params = {
            "orderBy": '"timestamp_ms"',
            "limitToLast": str(limit)
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data if data else {}
    except Exception:
        try:
            r2 = requests.get(url, timeout=12)
            r2.raise_for_status()
            data2 = r2.json()
            return data2 if data2 else {}
        except Exception:
            return {}

def get_latest_device_event_from_recent(device_id, recent_events):
    latest = None
    latest_ts = -1
    for k, v in (recent_events or {}).items():
        if not isinstance(v, dict): continue
        if v.get("device_id") != device_id: continue
        ts_num = None
        if "timestamp_ms" in v:
            try: ts_num = float(v["timestamp_ms"])
            except Exception: ts_num = None
        elif "timestamp_iso" in v:
            try:
                dt = datetime.datetime.fromisoformat(v["timestamp_iso"].replace("+05:30",""))
                ts_num = dt.timestamp() * 1000.0
            except Exception: ts_num = None
        if ts_num is None: continue
        if ts_num > latest_ts:
            latest_ts = ts_num
            latest = v
            latest["_fb_key"] = k
            latest["_fb_ts_ms"] = ts_num
    return latest

# ---------- Session state defaults ----------
if "results" not in st.session_state:
    st.session_state.results = {}
if "map_ready" not in st.session_state:
    st.session_state.map_ready = False
if "last_points" not in st.session_state:
    st.session_state.last_points = None
if "force_rerun" not in st.session_state:
    st.session_state.force_rerun = False
if "last_device_ts" not in st.session_state:
    st.session_state["last_device_ts"] = 0.0
if "last_device_event" not in st.session_state:
    st.session_state["last_device_event"] = None
if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False

# ---------- UI ----------
if not st.session_state.get("last_device_event"):
    st.session_state["last_device_event"] = {
        "latitude": 21.1458,
        "longitude": 79.0882,
        "gps_real": False
    }

st.markdown('<div class="app-title">Smart Safety Handbag 🛡️</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛠️ Configuration")
    try:
        df = load_data()
    except Exception as e:
        st.error("CSV load error: " + str(e))
        st.stop()

    mode = st.radio("Navigation Mode", ["Ward Demo (fast)", "Road Mode (real routing)"])
    input_type = st.radio("Input Source", ["Start & End Ward", "Start/End Coordinates"], index=1)

    with st.expander("📡 Device Connectivity", expanded=True):
        use_device_location = st.checkbox("Link Live Firebase Location", value=False)
        auto_poll = st.checkbox("Auto-poll updates", value=False)
        debug_fb = st.checkbox("Show raw data", value=False)
        
    st.markdown("---")
    st.markdown("### 🎨 Map Visuals")
    map_dark_mode = st.toggle("Enable Dark Mode Map", value=False)

    WARD_CSV = "data/nagpur_ward_centroids.csv"
    @st.cache_data
    def load_ward_centroids():
        if not os.path.exists(WARD_CSV):
            st.error("Missing nagpur_ward_centroids.csv")
            st.stop()
        wdf = pd.read_csv(WARD_CSV)
        return {row["Ward"]: (float(row["Latitude"]), float(row["Longitude"])) for _, row in wdf.iterrows()}

    ward_centroids = load_ward_centroids()
    ward_names = sorted(list(ward_centroids.keys()))

    device_event = None
    # Fix 2: Prevent auto-poll from overwriting ward mode
    if use_device_location and auto_poll and input_type != "Start & End Ward":
        node = fetch_latest_event_node(DEVICE_ID)
        if debug_fb: st.expander("Raw Firebase Node").write(node)
        if node:
            ts_ms = None
            if "timestamp_ms" in node:
                try: ts_ms = float(node["timestamp_ms"])
                except Exception: ts_ms = None
            elif "timestamp_iso" in node:
                try:
                    dt = datetime.datetime.fromisoformat(node["timestamp_iso"].replace("+05:30", ""))
                    ts_ms = dt.timestamp() * 1000.0
                except Exception: ts_ms = None
            if ts_ms is None:
                if st.session_state["last_device_event"] is None: st.session_state["last_device_event"] = node
            else:
                if ts_ms > float(st.session_state["last_device_ts"]):
                    st.session_state["last_device_ts"] = ts_ms
                    st.session_state["last_device_event"] = node

    if use_device_location and st.session_state.get("last_device_event"):
        device_event = st.session_state["last_device_event"]

    st.divider()
    st.markdown("### 📍 Location Details")
    if input_type == "Start & End Ward":
        start_ward = st.selectbox("Start Ward", ward_names)
        end_ward   = st.selectbox("End Ward", ward_names, index=1)

        start_lat, start_lon = ward_centroids[start_ward]
        end_lat, end_lon     = ward_centroids[end_ward]
    else:
        start_lat_default = float(device_event["latitude"]) if device_event and "latitude" in device_event else 21.145800
        start_lon_default = float(device_event["longitude"]) if device_event and "longitude" in device_event else 79.088200
        start_lat = st.number_input("Start latitude", value=float(start_lat_default), format="%.6f")
        start_lon = st.number_input("Start longitude", value=float(start_lon_default), format="%.6f")
        end_lat = st.number_input("End latitude", value=21.184000, format="%.6f")
        end_lon = st.number_input("End longitude", value=79.106000, format="%.6f")

    current_points = (float(start_lat), float(start_lon), float(end_lat), float(end_lon))

    time_input = st.time_input("Departure Time", value=datetime.datetime.now().time())
    dt = datetime.datetime.combine(datetime.date.today(), time_input)

    crime_artifact = load_artifact(CRIME_MODEL_PKL)
    risk_artifact = load_artifact(RISK_MODEL_PKL)
    
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1: run_button = st.button("Calculate", use_container_width=True, type="primary")
    with col_btn2: clear_button = st.button("Clear", use_container_width=True)

# Fix 1: Stop infinite loop logic
if run_button:
    st.session_state.last_points = current_points
    st.session_state.force_rerun = False

if clear_button:
    st.session_state.results = {}
    st.session_state.last_points = None
    st.session_state.force_rerun = False

device_event = device_event if 'device_event' in locals() and device_event is not None else st.session_state.get("last_device_event")

st.markdown('<div class="section-header">🔍 Predictive Analytics</div>', unsafe_allow_html=True)
col1, col2 = st.columns([1,1])

with col1:
    st.markdown('<div class="card card-risk">', unsafe_allow_html=True)
    if st.session_state.results.get("risk_label"):
        st.metric("Estimated Risk Level", st.session_state.results['risk_label'])
        with st.expander("Probability Breakdown"):
            st.dataframe(pd.DataFrame(st.session_state.results.get("risk_proba", {}).items(), columns=["Level", "Confidence"]), use_container_width=True)
    else: st.info("Run calculation for Risk Prediction")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card card-crime">', unsafe_allow_html=True)
    if st.session_state.results.get("crime_label"):
        st.metric("Dominant Crime Type", st.session_state.results['crime_label'])
        with st.expander("Probability Breakdown"):
            st.dataframe(pd.DataFrame(st.session_state.results.get("crime_proba", {}).items(), columns=["Type", "Confidence"]), use_container_width=True)
    else: st.info("Run calculation for Crime Prediction")
    st.markdown('</div>', unsafe_allow_html=True)

map_container = st.container()

if run_button or st.session_state.force_rerun:
    if crime_artifact is None or risk_artifact is None: st.error("Model artifacts missing.")
    else:
        with st.status("🧠 Analyzing data and routing...", expanded=True) as status:
            try:
                risk_label, risk_proba = predict_with_artifact(start_lat, start_lon, dt, risk_artifact)
                crime_label, crime_proba = predict_with_artifact(start_lat, start_lon, dt, crime_artifact)
                st.session_state.results.update({"risk_label": risk_label, "risk_proba": risk_proba, "crime_label": crime_label, "crime_proba": crime_proba})
            except Exception as e: st.error(f"Error: {e}")

            gh_coords = []
            try: gh_coords = get_route_graphhopper(start_lat, start_lon, end_lat, end_lon, api_key=DEFAULT_GH_API_KEY)
            except Exception: gh_coords = []

            route_safe_coords = []
            try:
                Gp, G_latlon = ensure_graph()
                orig_node = nearest_node_wrapper(G_latlon, start_lon, start_lat)
                dest_node = nearest_node_wrapper(G_latlon, end_lon, end_lat)
                has_safety = any("safety_weight" in d for _, _, d in Gp.edges(data=True))
                if has_safety:
                    route_safe_local = nx.shortest_path(Gp, orig_node, dest_node, weight="safety_weight")
                    route_safe_coords = route_coords_from_graph(G_latlon, route_safe_local)
            except Exception: route_safe_coords = []

            st.session_state.results.update({
                "gh_coords": gh_coords, "route_safe_coords": route_safe_coords,
                "map_center": ((start_lat + end_lat) / 2.0, (start_lon + end_lon) / 2.0)
            })
            st.session_state.map_ready = True
            status.update(label="✅ Analysis Complete!", state="complete")

res = st.session_state.results
if st.session_state.map_ready and res:
    mode_text = "Dark" if map_dark_mode else "Light"
    st.markdown(f'<div class="section-header">🗺️ Safety Map ({mode_text} Mode)</div>', unsafe_allow_html=True)
    with map_container:
        st.markdown('<div class="map-wrapper">', unsafe_allow_html=True)
        # Fix 3: Direct center calculation for freshness
        center = (
            (start_lat + end_lat) / 2.0,
            (start_lon + end_lon) / 2.0
        )
        tiles = "CartoDB dark_matter" if map_dark_mode else "OpenStreetMap"
        m = folium.Map(location=list(center), zoom_start=13, tiles=tiles)

        gh_coords = res.get("gh_coords", [])
        safe_coords = res.get("route_safe_coords", [])

        if gh_coords: folium.PolyLine(gh_coords, color="#3498db", weight=5, opacity=0.8, tooltip="Shortest").add_to(m)
        if safe_coords: folium.PolyLine(safe_coords, color="#2ecc71", weight=5, opacity=0.9, tooltip="Safest").add_to(m)

        start_pt = gh_coords[0] if gh_coords else (start_lat, start_lon)
        end_pt = gh_coords[-1] if gh_coords else (end_lat, end_lon)
        folium.Marker(start_pt, icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker(end_pt, icon=folium.Icon(color="red", icon="stop")).add_to(m)

        if device_event:
            try:
                dlat, dlng = float(device_event.get("latitude")), float(device_event.get("longitude"))
                folium.Marker([dlat, dlng], popup=f"Handbag: {DEVICE_ID}", icon=folium.Icon(color="purple", icon="briefcase", prefix='fa')).add_to(m)
            except: pass

        st_folium(m, width=1200, height=600, key="main_map", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed for Nagpur City Safety Awareness Dashboard")