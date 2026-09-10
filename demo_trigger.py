"""
demo_trigger.py
----------------
Command-line helper for your LIVE DEMO. Fires a "rainfall spike" at a
chosen village through the API and prints back the new risk level -
this is the moment your dashboard should visibly react on screen.

Usage:
  python demo_trigger.py UK01
  python demo_trigger.py UK01 --rainfall 190 --soil 90
"""

import argparse
import requests

API_URL = "http://127.0.0.1:8000"

parser = argparse.ArgumentParser(description="Trigger a sensor event for the demo")
parser.add_argument("village_id", help="Target Village ID (e.g. UK01 - UK07)")
parser.add_argument("--rainfall", type=float, help="Override rainfall (mm)")
parser.add_argument("--soil", type=float, help="Override soil moisture (0-100 %%)")
args = parser.parse_args()

payload = {}
if args.rainfall is not None:
    payload["rainfall"] = args.rainfall
if args.soil is not None:
    payload["soil_pct"] = args.soil

url = f"{API_URL}/trigger_event/{args.village_id}"
response = requests.post(url, json=payload)

if response.status_code == 200:
    print("Event triggered successfully:")
    print(response.json())
else:
    print(f"Failed ({response.status_code}): {response.text}")
