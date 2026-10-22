import urllib.request
import json

try:
    req = urllib.request.urlopen("https://api.gofile.io/servers")
    res = json.loads(req.read().decode())
    print("Gofile servers response:", res)
except Exception as e:
    print("Error querying gofile:", e)
