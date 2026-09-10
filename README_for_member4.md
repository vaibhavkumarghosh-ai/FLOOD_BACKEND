# Flash Flood Backend v2 — for Member 4

## What changed from the first version
- Wired to the REAL dataset (`flash_flood_features_uttarakhand.csv`, 7 Uttarakhand
  villages: UK01-UK07) instead of fake sample data.
- Model features now match Member 2's notebook exactly (soil moisture in
  m3/m3, slope in degrees, GSI landslide risk, etc.)
- Added SQLite logging (`flood_logs.db`) — every prediction (auto-simulated,
  manually triggered, or one-off) is saved with a timestamp. New `/logs`
  endpoint lets you build an alert-history panel.
- Risk is computed immediately at server startup now — `/status` returns
  real numbers on your very first call, not zeros.

## ⚠️ One thing to know before you build against this
Member 2's notebook was trained on a target column (`Historically_Flooded_Area`)
from a file that wasn't shared with me — only the live feature data was. So
`flood_model.pkl` in this package is a TEMPORARY model with the right feature
schema, not Member 2's real trained model. The prediction logic and API
shape will NOT change when the real model is swapped in — only the risk
scores' accuracy will improve. Safe to build your dashboard against this now.

## Setup
```bash
pip install fastapi uvicorn scikit-learn joblib pandas numpy requests
python train_model_temp.py     # creates flood_model.pkl (temporary)
uvicorn app:app --reload --port 8000
```
Visit `http://127.0.0.1:8000/docs` for a free interactive test page.

## Endpoints for Member 4's dashboard

**`GET /status`** — poll this every few seconds. Returns a list of all 7
villages with their current state:
```json
{
  "village_id": "UK01",
  "village_name": "Kedarnath_Valley",
  "district": "Rudraprayag",
  "lat": 30.73, "lon": 79.06,
  "elevation_m": 3584, "slope_deg": 38,
  "gsi_risk_label": "Very High Risk",
  "rainfall_mm": 133.7,
  "soil_moisture_pct": 65.4,
  "risk_score": 85.6,
  "risk_level": "Very High",       // one of: Negligible/Low/Medium/High/Very High
  "estimated_lead_time_hrs": 2.0,  // hours of warning before impact
  "last_updated": 1789064386.5
}
```
Use `risk_level` to color-code your map markers (e.g. green/yellow/orange/
red/dark-red), and `estimated_lead_time_hrs` to show "X hours to evacuate"
on each village's popup.

**`GET /villages`** — same shape, useful to draw the initial map before you
start polling `/status`.

**`GET /logs?limit=50`** — recent prediction history across all villages.
Add `&village_id=UK01` to filter to one village. Good for an "Alert Log" /
"Recent Events" panel showing a scrolling history.

**`POST /trigger_event/{village_id}`** — used during the live demo (Member 3
runs this from the terminal), but you can also wire a "Simulate Flood"
button in your own dashboard that calls this directly if you want the demo
triggerable from the UI itself:
```json
POST /trigger_event/UK01
{ "rainfall": 190, "soil_pct": 90 }
```

## Notes on data
- 7 villages total: UK01 (Kedarnath_Valley) through UK07 (Govindghat_Zone)
- Background simulator nudges one random village's rainfall/soil every 8
  seconds automatically — your dashboard should look "alive" on its own
  without anyone touching anything.
