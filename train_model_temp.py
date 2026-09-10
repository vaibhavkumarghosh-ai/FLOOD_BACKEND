"""
train_model_temp.py
--------------------
IMPORTANT CONTEXT:
Member 2's notebook (flood.ipynb) trains on a file called 'flashflood.csv'
(loaded from their Google Drive) that has a target column called
'Historically_Flooded_Area'. That training file was NOT uploaded here -
only the live feature data (flash_flood_features_uttarakhand.csv) was,
which has no target column. The notebook also never calls joblib.dump(),
so no .pkl file exists to hand off yet.

This script is a STAND-IN so your API has a working flood_model.pkl today.
It builds a synthetic training set using the EXACT same feature columns
and RandomForestClassifier settings as Member 2's notebook, so the file
format is a drop-in match.

>>> REPLACE THIS THE MOMENT MEMBER 2 GIVES YOU A REAL flood_model.pkl <<<
Tell Member 2 to add this one line at the end of their notebook:
    import joblib
    joblib.dump(model, "flood_model.pkl")
Then just drop their file into this folder, same filename. No code
elsewhere needs to change, AS LONG AS their feature column order matches
the FEATURES list below exactly (it does, since it's copied from their notebook).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)

FEATURES = [
    "SMAP_Soil_Moisture_m3_m3",
    "Surface_Temp_C",
    "Latitude",
    "Longitude",
    "SRTM_Mean_Elevation_m",
    "SRTM_Calculated_Slope_deg",
    "District_Rainfall_Sep2026_mm",
    "GSI_Landslide_Risk",   # numeric-encoded version of GSI_Landslide_Susceptibility
]

# ---- Generate synthetic training rows spanning realistic value ranges ----
n = 300
df = pd.DataFrame({
    "SMAP_Soil_Moisture_m3_m3": np.random.uniform(0.1, 0.5, n),
    "Surface_Temp_C": np.random.uniform(10, 30, n),
    "Latitude": np.random.uniform(29.5, 31.0, n),
    "Longitude": np.random.uniform(78.5, 80.5, n),
    "SRTM_Mean_Elevation_m": np.random.uniform(800, 3600, n),
    "SRTM_Calculated_Slope_deg": np.random.uniform(10, 45, n),
    "District_Rainfall_Sep2026_mm": np.random.uniform(20, 200, n),
    "GSI_Landslide_Risk": np.random.choice([0, 1, 2, 3], n),  # Low/Moderate/High/Very High
})

# Synthetic flood label - higher rainfall + wet soil + steep slope + high GSI risk -> more likely flooded
flood_probability = (
    (df["District_Rainfall_Sep2026_mm"] / 200) * 0.4
    + (df["SMAP_Soil_Moisture_m3_m3"] / 0.5) * 0.25
    + (df["SRTM_Calculated_Slope_deg"] / 45) * 0.15
    + (df["GSI_Landslide_Risk"] / 3) * 0.2
)
df["Historically_Flooded_Area"] = (flood_probability > 0.55).astype(int)

# ---- Train (same hyperparameters as Member 2's notebook cell 7) ----
X = df[FEATURES]
y = df["Historically_Flooded_Area"]

model = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=4,
    min_samples_leaf=2, random_state=50,
)
model.fit(X, y)

joblib.dump(model, "flood_model.pkl")
print("Saved TEMPORARY model -> flood_model.pkl")
print("Classes:", model.classes_)
print("Feature order (must match Member 2's real model):", FEATURES)
