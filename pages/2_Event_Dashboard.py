import streamlit as st
import requests
import pandas as pd

FIREBASE_EVENTS_URL = "https://smart-safety-handbag-default-rtdb.asia-southeast1.firebasedatabase.app/events.json"

st.set_page_config(page_title="Event Dashboard", layout="wide")
st.title("📡 Device Event Dashboard")

@st.cache_data(ttl=10)
def load_events():
    try:
        resp = requests.get(FIREBASE_EVENTS_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception as e:
        st.error(f"Firebase fetch error: {e}")
        return pd.DataFrame()

    rows = []
    for key, v in data.items():
        if not isinstance(v, dict):
            continue
        rows.append({
            "ID": key,
            "Type": v.get("event_type", ""),
            "Source": v.get("source", ""),
            "Severity": v.get("severity", ""),
            "Latitude": v.get("latitude"),
            "Longitude": v.get("longitude"),
            "Timestamp": v.get("timestamp_iso", ""),
            "Device": v.get("device_id", "")
        })

    return pd.DataFrame(rows)

df = load_events()

if df.empty:
    st.warning("No events yet from device.")
    st.stop()

st.sidebar.header("Filters")

severity = st.sidebar.selectbox("Severity", ["All"] + sorted(df["Severity"].unique().tolist()))
if severity != "All":
    df = df[df["Severity"] == severity]

st.dataframe(df, use_container_width=True)

st.subheader("📍 Event Map")
locs = df.dropna(subset=["Latitude", "Longitude"])
if not locs.empty:
    st.map(locs.rename(columns={"Latitude": "lat", "Longitude": "lon"}))
else:
    st.info("No GPS data available.")
