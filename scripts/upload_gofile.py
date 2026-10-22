import requests
import json
import time

print("Fetching available Gofile server...")
srv_res = requests.get("https://api.gofile.io/servers").json()
server = srv_res["data"]["servers"][0]["name"]
print(f"Using server: {server}")

upload_url = f"https://{server}.gofile.io/contents/uploadfile"
file_path = "/content/newsseek_vector_stores.zip"

print(f"Starting upload of {file_path}...")
t0 = time.time()
with open(file_path, "rb") as f:
    res = requests.post(upload_url, files={"file": f})

elapsed = time.time() - t0
print(f"Upload finished in {elapsed:.1f}s. Response status: {res.status_code}")
data = res.json()
print("Response JSON:")
print(json.dumps(data, indent=2))
