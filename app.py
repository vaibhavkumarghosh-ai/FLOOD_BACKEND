import threading, time, random, sqlite3, joblib
from datetime import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = "flood_model.pkl"
DATA_PATH = "flash_flood_features_uttarakhand.csv"
DB_PATH = "flood_logs.db"

FEATURES = [
    "SMAP_Soil_Moisture_m3_m3", "Surface_Temp_C", "Latitude", "Longitude",
    "SRTM_Mean_Elevation_m", "SRTM_Calculated_Slope_deg",
    "District_Rainfall_Sep2026_mm", "GSI_Landslide_Risk"
]

app = FastAPI(title="SIH 2026 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

lock = threading.Lock()
live_state = {}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS prediction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, village_id TEXT,
        village_name TEXT, rainfall_mm REAL, soil_moisture_pct REAL,
        risk_score REAL, risk_level TEXT, lead_time_hrs REAL, source TEXT)""")
    conn.commit()
    conn.close()

def log_prediction(village_id, village_name, rainfall, soil_pct, risk_score, risk_level, lead_time, source):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO prediction_log (timestamp, village_id, village_name, rainfall_mm, soil_moisture_pct, risk_score, risk_level, lead_time_hrs, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                 (datetime.now().isoformat(timespec="seconds"), village_id, village_name, rainfall, soil_pct, risk_score, risk_level, lead_time, source))
    conn.commit()
    conn.close()

init_db()

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

def encode_gsi(label) -> float:
    if isinstance(label, (int, float)): return float(label)
    l = str(label).lower()
    if "very high" in l: return 3.0
    if "high" in l: return 2.0
    if "moderate" in l or "medium" in l: return 1.0
    return 0.0

try:
    df_init = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    if "GSI_Landslide_Risk" not in df_init.columns:
        df_init["GSI_Landslide_Risk"] = df_init["GSI_Landslide_Susceptibility"].map(encode_gsi)
    latest = df_init.sort_values("Date").groupby("Village_ID").tail(1)
    for _, row in latest.iterrows():
        v_id = str(row["Village_ID"]).upper()
        soil_raw = float(row["SMAP_Soil_Moisture_m3_m3"])
        live_state[v_id] = {
            "village_id": v_id, "village_name": row["Village_Name"], "district": row["District"],
            "lat": float(row["Latitude"]), "lon": float(row["Longitude"]),
            "elevation_m": float(row["SRTM_Mean_Elevation_m"]), "slope_deg": float(row["SRTM_Calculated_Slope_deg"]),
            "gsi_risk_numeric": float(row["GSI_Landslide_Risk"]), "rainfall_mm": float(row["District_Rainfall_Sep2026_mm"]),
            "soil_moisture_raw": soil_raw, "soil_moisture_pct": round(min(100.0, (soil_raw / 0.5) * 100.0), 1),
            "surface_temp_c": float(row["Surface_Temp_C"]), "risk_score": 0.0, "risk_level": "Negligible",
            "estimated_lead_time_hrs": 12.0, "last_updated": time.time()
        }
except Exception as e:
    print("Data load error:", e)

def run_prediction(vdata: dict):
    rainfall = vdata.get("rainfall_mm", 0.0)
    soil_pct = vdata.get("soil_moisture_pct", 0.0)
    ml_comp = min(100.0, (rainfall * 0.4) + (soil_pct * 0.4))
    if model is not None:
        try:
            X = pd.DataFrame([{
                "SMAP_Soil_Moisture_m3_m3": vdata.get("soil_moisture_raw", 0.3),
                "Surface_Temp_C": vdata.get("surface_temp_c", 20.0),
                "Latitude": vdata.get("lat", 30.0), "Longitude": vdata.get("lon", 79.0),
                "SRTM_Mean_Elevation_m": vdata.get("elevation_m", 1500.0),
                "SRTM_Calculated_Slope_deg": vdata.get("slope_deg", 15.0),
                "District_Rainfall_Sep2026_mm": rainfall,
                "GSI_Landslide_Risk": vdata.get("gsi_risk_numeric", 1.0)
            }])[FEATURES]
            proba = model.predict_proba(X)[0]
            ml_comp = float(proba[1]) * 100.0 if len(model.classes_) == 2 else float(np.dot(proba, np.linspace(0, 100, len(model.classes_))))
        except Exception:
            pass
    risk_score = round(ml_comp, 1)
    if rainfall >= 140.0 or risk_score >= 85.0: risk_level = "Very High"
    elif rainfall >= 90.0 or risk_score >= 65.0: risk_level = "High"
    elif rainfall >= 45.0 or risk_score >= 45.0: risk_level = "Medium"
    elif rainfall >= 15.0 or risk_score >= 20.0: risk_level = "Low"
    else: risk_level = "Negligible"
    return risk_score, risk_level

def apply_prediction(vdata: dict, source: str):
    risk_score, risk_level = run_prediction(vdata)
    lead_time = 2.0 if risk_level == "Very High" else (4.0 if risk_level == "High" else 8.0)
    vdata["risk_score"] = risk_score
    vdata["risk_level"] = risk_level
    vdata["estimated_lead_time_hrs"] = lead_time
    vdata["last_updated"] = time.time()
    log_prediction(vdata["village_id"], vdata.get("village_name", ""), vdata.get("rainfall_mm", 0.0), vdata.get("soil_moisture_pct", 0.0), risk_score, risk_level, lead_time, source)
    return risk_score, risk_level, lead_time

class TriggerPayload(BaseModel):
    rainfall: float | None = None
    soil_pct: float | None = None

@app.get("/")
def read_root(): return {"status": "ok", "villages_loaded": len(live_state)}

@app.get("/villages")
def get_villages():
    with lock: return list(live_state.values())

@app.post("/trigger_event/{village_id}")
def trigger_event(village_id: str, payload: TriggerPayload):
    target_id = village_id.upper()
    with lock:
        if target_id not in live_state:
            raise HTTPException(status_code=404, detail=f"Village ID '{village_id}' not found.")
        v = live_state[target_id]
        if payload.rainfall is not None: v["rainfall_mm"] = payload.rainfall
        if payload.soil_pct is not None:
            v["soil_moisture_pct"] = payload.soil_pct
            v["soil_moisture_raw"] = round((payload.soil_pct / 100.0) * 0.5, 3)
        risk_score, risk_level, lead_time = apply_prediction(v, "manual_trigger")
        return {"message": f"Triggered {v['village_name']} ({target_id})", "risk_level": risk_level, "risk_score": risk_score, "estimated_lead_time_hrs": lead_time}

@app.get("/logs")
def get_logs(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM prediction_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
