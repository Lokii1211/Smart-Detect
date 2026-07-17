"""Seed location + start camera for dashboard live stream."""
import requests

API = "http://localhost:8000"

# 1. Login
r = requests.post(f"{API}/auth/login", json={"username": "operator", "password": "smartOp2024"})
token = r.json()["access_token"]
h = {"Authorization": f"Bearer {token}"}
print("Token: OK")

# 2. Check locations
locs = requests.get(f"{API}/locations", headers=h).json()
loc_ids = [l["id"] for l in locs]
print(f"Locations: {loc_ids}")

# 3. Create location if needed
if "LOC-001" not in loc_ids:
    r = requests.post(f"{API}/locations", json={"id": "LOC-001", "name": "Main Entrance", "type": "entrance"}, headers=h)
    print(f"Created location: {r.status_code}")

# 4. Check cameras
cams = requests.get(f"{API}/cameras", headers=h).json()
print(f"Camera groups: {len(cams)}")
for g in cams:
    cam_ids = [c["id"] for c in g["cameras"]]
    print(f"  {g['location_name']}: {cam_ids}")

# 5. Check status
st = requests.get(f"{API}/camera/status").json()
print(f"Active: {st['active_cameras']}/{st['total_cameras']}")

# 6. Start CAM-001
if st["active_cameras"] == 0:
    r = requests.post(f"{API}/camera/start", json={"camera_id": "CAM-001"}, headers=h)
    print(f"Start camera: {r.status_code} {r.json()}")
else:
    print("Camera already running")

# 7. Verify stream
st2 = requests.get(f"{API}/camera/status").json()
print(f"Final status: {st2['active_cameras']}/{st2['total_cameras']} active")
