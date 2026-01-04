import streamlit as st
import requests
import time
import datetime
import joblib
from dotenv import load_dotenv
import os
import threading

load_dotenv()

FIREBASE_URL = os.getenv("FIREBASE_URL")
DEVICE_ID = os.getenv("DEVICE_ID", "handbag_001")
EVENTS_PATH = "events"

RISK_MODEL_PKL = "models/risk_model.pkl"
CRIME_MODEL_PKL = "models/crime_type_model.pkl"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

risk_artifact = load_model(RISK_MODEL_PKL)
crime_artifact = load_model(CRIME_MODEL_PKL)

stop_flag = False

def prepare_features(event):
    return [[
        0,
        float(event.get("latitude", 0)),
        float(event.get("longitude", 0)),
        datetime.datetime.now().hour,
        0,
        0
    ]]

def predict_event(event):
    X = prepare_features(event)
    r = risk_artifact["model"].predict(X)[0]
    c = crime_artifact["model"].predict(X)[0]
    return str(r), str(c)

def write_worker_heartbeat():
    try:
        hb = {"ts": int(time.time() * 1000)}
        requests.put(f"{FIREBASE_URL}/system_status/ml_worker_heartbeat.json", json=hb, timeout=5)
    except:
        pass

def polling_loop():
    global stop_flag
    while not stop_flag:
        try:
            url = f"{FIREBASE_URL}/{EVENTS_PATH}.json"
            events = requests.get(url, timeout=10).json()

            for key, ev in (events or {}).items():
                if ev is None or ev.get("processed"):
                    continue

                risk, crime = predict_event(ev)
                res = {
                    "risk": risk,
                    "crime": crime,
                    "predicted_at": datetime.datetime.now().isoformat()
                }

                requests.put(f"{FIREBASE_URL}/{EVENTS_PATH}/{key}/ml_result.json", json=res)
                requests.put(f"{FIREBASE_URL}/{EVENTS_PATH}/{key}/processed.json", json=True)

            write_worker_heartbeat()

        except:
            pass

        time.sleep(5)

st.set_page_config(page_title="ML Firebase Worker", layout="wide")
st.title("🧠 ML Firebase Worker")

if "worker" not in st.session_state:
    st.session_state.worker = None
    st.session_state.running = False

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Worker"):
        if not st.session_state.running:
            stop_flag = False
            t = threading.Thread(target=polling_loop, daemon=True)
            t.start()
            st.session_state.worker = t
            st.session_state.running = True
            st.success("Worker started")

with col2:
    if st.button("⏹ Stop Worker"):
        stop_flag = True
        st.session_state.running = False
        st.warning("Worker stopped")

st.write("Status:", "Running" if st.session_state.running else "Stopped")
