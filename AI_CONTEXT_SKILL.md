---
name: Macro Forecaster Infrastructure & Knowledge Base
description: Comprehensive documentation, IP addresses, architectural decisions, and bug-fixes for the Macro Forecaster project.
---

# Macro Forecaster - AI Knowledge Base & Context Skill

This document contains all the critical context, infrastructure details, IP addresses, tokens, and historical bug fixes for the Macro Forecaster project. **Any future AI agents should read this file before making modifications to the infrastructure or architecture.**

## 1. Project Architecture

The Macro Forecaster is an institutional-grade macroeconomic predictive dashboard utilizing XGBoost and FinBERT to forecast asset directional edge.

- **Frontend**: Next.js (React), TailwindCSS, Recharts, hosted on **Vercel**.
- **Backend API**: Python 3.12, FastAPI, Uvicorn, hosted on a remote Ubuntu server.
- **Machine Learning**: XGBoost (tabular data), FinBERT/RoBERTa (NLP Sentiment), Yahoo Finance (yfinance) for real-time spot prices.
- **Network Bridge**: Cloudflare Quick Tunnels (`cloudflared`) exposing the backend to the public internet.

## 2. Infrastructure & IP Addresses

### Remote Backend Server (Laravel Server)
- **IP Address**: `192.248.43.132`
- **SSH Access**: Connect via `ssh laravel-server` (Configured in local `~/.ssh/config`).
- **Project Path**: `/root/macro-forecaster/backend`
- **Virtual Environment**: `/root/macro-forecaster/backend/venv` (Python 3.12)
- **Service Port**: `8000` (FastAPI/Uvicorn)

### Commands for Managing the Backend
To pull latest code and restart the production FastAPI server in the background:
```bash
ssh laravel-server "cd /root/macro-forecaster/backend && git pull && fuser -k 8000/tcp; nohup venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
```

To restart the Cloudflare Tunnel (if the current `.trycloudflare.com` URL dies):
```bash
ssh laravel-server "fuser -k 8000/tcp; nohup cloudflared port-forward --url http://localhost:8000 > tunnel.log 2>&1 &"
# Then read the tunnel.log to get the new URL:
ssh laravel-server "cat tunnel.log | grep trycloudflare"
```

## 3. Critical Errors, Bugs, and Historical Fixes

### Error 1: 502 Bad Gateway / Network Unreachable
- **Symptoms**: The frontend could not reach the FastAPI server.
- **Root Cause**: The Python process was bound to `127.0.0.1` (localhost only) or the Cloudflare tunnel was misconfigured.
- **Fix Applied**: Forced Uvicorn to bind to `0.0.0.0` allowing external connections: `uvicorn main:app --host 0.0.0.0 --port 8000`. 

### Error 2: Vercel API Proxy Blocked (503 Service Unavailable / CORS Errors)
- **Symptoms**: The Next.js frontend worked locally, but when deployed to Vercel, API requests to the Cloudflare tunnel failed silently or threw 503 errors.
- **Root Cause**: Next.js originally used a server-side API proxy (`/api/proxy`). When Vercel's AWS IP addresses attempted to hit the `trycloudflare.com` tunnel, Cloudflare's Bot Protection immediately blocked the AWS datacenter IPs.
- **Fix Applied (CRITICAL)**: 
  1. Deleted the Next.js API proxy route.
  2. Bypassed Vercel completely by forcing the React client-side browser to fetch directly from the Cloudflare URL.
  3. Added `CORSMiddleware(allow_origins=["*"])` to `backend/main.py` so the browser wouldn't block the cross-origin request.

### Error 3: Race Condition on Dashboard Load
- **Symptoms**: The frontend would flash a "Network Error" on hard refresh before successfully connecting.
- **Root Cause**: The `backendUrl` state was initialized with a hard-coded, dead Localtunnel URL (`three-snakes-sleep.loca.lt`). The component fetched this dead URL *before* it could load the user's active Cloudflare URL from `localStorage`.
- **Fix Applied**: Initialized the default URL to `""` and added an `isInitialized` flag to ensure the `fetchData()` hook only fires *after* the localStorage is read. Added auto-injection of `https://` if the user forgets to type it.

### Error 4: Probability Math Scaling & Flat Charts
- **Symptoms**: Directional Probabilities were perpetually stuck at 51%, and the visual forecast chart line was flat.
- **Root Cause**: The ML engine outputs predicted log returns (which are tiny basis points, e.g., 0.0001). The sigmoid scaling formula used a multiplier of `50`, which mathematically squashed all realistic daily returns into 0.501 (51%). The chart divided these tiny fractions over 7 days, resulting in zero visual movement.
- **Fix Applied**: 
  1. Increased the backend sigmoid multiplier in `main.py` from `50` to `1000`.
  2. Increased the frontend visual chart multiplier in `BeautifulChart.tsx` to explicitly curve the line to visually represent the directional edge (Bullish/Bearish) against 1% historical volatility.

## 4. Environment & Deployment Workflow

1. **Local Development**: Code modifications are made locally on the Windows machine (`e:/AI PROJECT FOR PROFILE/`).
2. **Version Control**: Changes are committed and pushed to GitHub (`wathsalarangajeewamail4-lab/macro-forecaster`).
3. **Frontend Deployment**: Vercel automatically detects the GitHub push and rebuilds the Next.js frontend.
4. **Backend Deployment**: Must SSH into `192.248.43.132` (`laravel-server`), run `git pull`, kill the existing process on port 8000, and run `nohup uvicorn` again.

---
*Document maintained by Antigravity AI.*
