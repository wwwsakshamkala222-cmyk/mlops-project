# BEAST v2.0 — Clinical Decision Support System

A production-grade hospital readmission prediction frontend.

## File Structure

```
beast-cdss/
├── index.html     ← Main HTML (structure + layout)
├── style.css      ← All styling (Dark Slate/Emerald theme)
├── app.js         ← All logic (vitals, prediction, rendering)
└── README.md      ← This file
```

## Quick Start

1. Open `index.html` in any modern browser — no build step required.
2. Demo Mode is ON by default (local prediction, no backend needed).
3. To connect to FastAPI backend: click **"Live Mode"** in the top-right nav.

## Connecting to FastAPI

The API call is in `app.js` → `callPredictAPI()`.

Your backend must accept:
```json
POST http://localhost:8000/predict
{
  "patient_id": "PT-2024-0831",
  "age": 58,
  "gender": "Male",
  "diagnosis": "Sepsis",
  "medication": "Heparin",
  "vitals": [97.0, 78.0, 105.0, 120.0, 80.0, 36.6, 16.0, 0.9]
}
```

And return either:
```json
{ "risk_score": 0.72 }          // 0–1 float, auto-converted to %
{ "risk_score": 72.0 }          // 0–100 float, used directly
{ "risk_score": 72.0, "feature_importance": [...] }  // optional
```

## Features

- Dark Slate/Emerald clinical theme
- Sidebar: Patient ID, Age, Gender, Ward, Physician info
- 3-tab layout: Model Inputs / Results / History
- 8 biomarker sliders with real-time safe/danger indicators
- Radial arc gauge with 3 color zones
- Priority badge (Stable / Elevated / Critical)
- Feature importance reasoning cards
- Prediction history table
- Demo mode + Live API toggle
- Graceful API error handling (timeout, connection, HTTP errors)
- Fully responsive (mobile-friendly)

## Error Handling

`callPredictAPI()` handles:
- `ConnectionError` — backend not running
- `TimeoutError` — server too slow (8s limit)
- `HTTPError` — 4xx / 5xx status codes
- `JSONDecodeError` — malformed response
- Score normalization — accepts 0–1 or 0–100 formats
