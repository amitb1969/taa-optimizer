# Global TAA Optimizer

Single-file HTML tactical asset allocation optimizer for the DL Partners TAA1 universe.
Correlations, volatilities, and latest prices are refreshed daily from Yahoo Finance
by a GitHub Action that runs `taa_data.py` and commits `taa_data.json`.

## Files

- `index.html` — the optimizer (all CSS/JS inline, no build step).
- `taa_data.py` — data feed: fetches prices, Ledoit-Wolf shrinkage, Higham PD correction,
  writes `taa_data.json`.
- `taa_data.json` — generated artifact, committed by the Action.
- `.github/workflows/update-data.yml` — daily cron (23:30 UTC Mon–Fri) + manual trigger.

On page load, `index.html` fetches `taa_data.json` from the same directory.
If present → live data. If missing → hardcoded fallback correlations.

## Run locally

```
pip install -r requirements.txt
python taa_data.py
# open index.html in a browser (needs to be served, e.g. `python -m http.server`)
```

A `file://` open will block `fetch()` — serve over HTTP:

```
python -m http.server 8000
# then visit http://localhost:8000
```

## Deploy

**Vercel** (recommended — matches the static-site stack):
1. Push this repo to GitHub.
2. Import into Vercel → accept the static-site defaults → deploy.

**GitHub Pages**:
1. Push to GitHub.
2. Repo → Settings → Pages → Source: `main` / root → Save.

Either way, the daily Action commits a fresh `taa_data.json`, and the next page load
picks it up.

## First run

The Action needs to run once before `taa_data.json` exists in the repo. Trigger it
manually from the Actions tab (`Run workflow`) after the first push, or run
`python taa_data.py` locally and commit the resulting JSON.
