# UniFi API cheatsheet — Delilah LA

```
CLOUD  api.ui.com          works from anywhere    NO VIDEO (endpoint doesn't exist)
LOCAL  10.0.14.10          needs venue network    VIDEO ✅
```

Load creds first (never echo them):

```bash
set -a; . ~/.env; set +a
CID='9C05D651C70F0000000007F029BD00000000085B1BBA0000000065D3A543:1358216062'
NVR=10.0.14.10
```

---

## A. CLOUD — works from India right now

```bash
# all 13 H.Wood consoles
curl -s -H "X-API-KEY: $UNIFI_API_KEY" https://api.ui.com/v1/hosts | jq '.data[].reportedState.name'

# one console
curl -s -H "X-API-KEY: $UNIFI_API_KEY" "https://api.ui.com/v1/hosts/$CID" | jq

# ALL CAMERAS, whole fleet — 349 devices. this is the useful one.
curl -s -H "X-API-KEY: $UNIFI_API_KEY" "https://api.ui.com/v1/devices?pageSize=500" \
  | jq '.data[] | select(.hostName|test("DLH LA")) | .devices[] | select(.productLine=="protect") | {name,model,ip}'

# network sites
curl -s -H "X-API-KEY: $UNIFI_API_KEY" https://api.ui.com/v1/sites | jq

# WAN performance
curl -s -H "X-API-KEY: $UNIFI_API_KEY" https://api.ui.com/v1/isp-metrics/5m | jq '.data[0].periods[-1].data.wan'
```

### Cloud dead ends — all tested, all fail

```bash
# 403 "user is not the owner of this host"  (personal key; org key or owner would fix)
curl -s -H "X-API-KEY: $UNIFI_API_KEY" \
  "https://api.ui.com/v1/connector/consoles/$CID/protect/integration/v1/cameras"

# 401 — Protect API keys are LOCAL-ONLY, the cloud does not know them
curl -s -H "X-API-KEY: $UNIFI_PROTECT_API_KEY" \
  "https://api.ui.com/v1/connector/consoles/$CID/protect/integration/v1/cameras"
```

Even with ownership: the connector caps responses at **10 MB** and times out at **25 s**,
and the Protect integration API has **no export endpoint at all**.

---

## B. LOCAL — needs a route to 10.0.14.10 (VPN / Tailscale / on-site)

### 1. Login (session cookie + CSRF)

```bash
curl -sk -c cookies.txt -D headers.txt \
  -X POST "https://$NVR/api/auth/login" \
  -H 'Content-Type: application/json' \
  -d "{\"username\":\"$UNIFI_NVR_USERNAME\",\"password\":\"$UNIFI_NVR_PASSWORD\"}"

CSRF=$(grep -i x-csrf-token headers.txt | awk '{print $2}' | tr -d '\r')
```

### 2. Camera list

```bash
curl -sk -b cookies.txt -H "x-csrf-token: $CSRF" \
  "https://$NVR/proxy/protect/api/bootstrap" | jq '.cameras[] | {id,name,state}'
```

### 3. ⭐ EXPORT VIDEO — the one that matters

```bash
# Jul 27 2026, 17:00-17:10 America/Los_Angeles  (channel 2 = low res, right for CV)
START=$(date -d '2026-07-27 17:00:00 America/Los_Angeles' +%s000)
END=$(date -d '2026-07-27 17:10:00 America/Los_Angeles' +%s000)

curl -sk -b cookies.txt -H "x-csrf-token: $CSRF" \
  "https://$NVR/proxy/protect/api/video/export?camera=<CAM_ID>&start=$START&end=$END&channel=2" \
  -o clip.mp4
```

### 4. Integration API (Protect API key instead of session)

```bash
curl -sk -H "X-API-KEY: $UNIFI_PROTECT_API_KEY" \
  "https://$NVR/proxy/protect/integration/v1/cameras" | jq

curl -sk -H "X-API-KEY: $UNIFI_PROTECT_API_KEY" \
  "https://$NVR/proxy/protect/integration/v1/cameras/<CAM_ID>/snapshot" -o snap.jpg
```

### 5. Live stream

```bash
curl -sk -H "X-API-KEY: $UNIFI_PROTECT_API_KEY" -H 'Content-Type: application/json' \
  -X POST "https://$NVR/proxy/protect/integration/v1/cameras/<CAM_ID>/rtsps-stream" \
  -d '{"qualities":["low"]}'
# -> {"low":"rtsps://10.0.14.10:7441/<alias>?enableSrtp"}

ffmpeg -rtsp_transport tcp -i "rtsps://..." -c copy \
       -f segment -segment_time 600 seg_%04d.mp4
```

---

## C. UNTESTED — the web relay

The Protect **website** streams video to you in India through Ubiquiti's relay.
That is a third system, separate from api.ui.com, and it has never been tested.

Capture it: Chrome F12 -> Network -> Preserve log -> Playback -> Download,
then find the large mp4 request and copy it as cURL.

If replayable, this is remote video with no VPN and nobody on site.

---

## Summary

| Want | Endpoint | Works from India? |
|---|---|---|
| Camera inventory | `GET api.ui.com/v1/devices` | ✅ now |
| Console list | `GET api.ui.com/v1/hosts` | ✅ now |
| Snapshot | `.../integration/v1/cameras/{id}/snapshot` | ❌ local only |
| **Recorded video** | `.../protect/api/video/export` | ❌ local only |
| Live stream | `.../rtsps-stream` + ffmpeg | ❌ local only |
| Recorded video via website | unknown | ❓ untested |
