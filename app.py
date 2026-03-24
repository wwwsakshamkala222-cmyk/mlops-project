"""
╔══════════════════════════════════════════════════════════════╗
║         MedEdge Risk API — FastAPI Server                    ║
║         Hospital Readmission Prediction · Production         ║
╚══════════════════════════════════════════════════════════════╝

THE KEY PRINCIPLE:
  ✅ Model, scaler, and encoders are loaded ONCE at startup
  ✅ Each /predict request does ONLY the tensor math — nothing else
  ✅ ~1-5ms per prediction instead of ~200-500ms

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ══════════════════════════════════════════════════════════════
#  CONFIG  (must match your training CONFIG exactly)
# ══════════════════════════════════════════════════════════════
MODEL_PATH   = "mededge_v4_best.pth"
SCALER_PATH  = "scaler.pkl"
LE_DIAG_PATH = "le_diag.pkl"
LE_MED_PATH  = "le_med.pkl"

VITALS_COLS = [
    "Blood_Glucose", "Creatinine", "Hemoglobin", "WBC",
    "Heart_Rate", "Blood_Pressure_Systolic", "SpO2", "Temperature"
]

RISK_THRESHOLD = 0.5   # tune this: lower = catch more at-risk patients

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mededge")


# ══════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE  (must be identical to train_mededge_v4.py)
# ══════════════════════════════════════════════════════════════
class SafeLabelEncoder:
    def __init__(self):
        self.classes_ = np.array([])
        self.class_to_idx = {}
    def transform(self, y):
        return np.array([self.class_to_idx.get(x, 0) for x in y])
    
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, out_dim)
        self.bn1  = nn.BatchNorm1d(out_dim)
        self.fc2  = nn.Linear(out_dim, out_dim)
        self.bn2  = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = torch.relu(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        return torch.relu(out + residual)


class MedEdgeV4(nn.Module):
    def __init__(self, num_diags, num_meds, num_vitals, cfg):
        super().__init__()
        ed, em, dr = cfg["diag_emb_dim"], cfg["med_emb_dim"], cfg["dropout"]
        hd = cfg["hidden_dims"]

        self.diag_emb   = nn.Embedding(num_diags, ed, padding_idx=0)
        self.med_emb    = nn.Embedding(num_meds,  em, padding_idx=0)
        self.diag_head  = nn.Sequential(nn.Linear(ed, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dr))
        self.med_head   = nn.Sequential(nn.Linear(em, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dr))
        self.vital_head = nn.Sequential(nn.Linear(num_vitals, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dr))

        trunk_in, layers, prev = 96, [], 96
        for h in hd:
            layers.append(ResidualBlock(prev, h, dr))
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.out   = nn.Linear(prev, 1)

    def forward(self, diag, med, vitals):
        x = torch.cat([
            self.diag_head(self.diag_emb(diag)),
            self.med_head(self.med_emb(med)),
            self.vital_head(vitals)
        ], dim=1)
        return self.out(self.trunk(x))


# ══════════════════════════════════════════════════════════════
#  GLOBAL STATE  — loaded once, shared across all requests
# ══════════════════════════════════════════════════════════════
# This is the RIGHT way.
# These objects live in RAM for the lifetime of the server process.
# Every request reads from memory — zero disk I/O per prediction.

class ModelState:
    model    : MedEdgeV4   = None
    scaler                 = None
    le_diag                = None
    le_med                 = None
    device   : torch.device = None
    cfg      : dict        = None

state = ModelState()


# ══════════════════════════════════════════════════════════════
#  LIFESPAN  — startup & shutdown events
# ══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ───────────────────────────────────────────────
    logger.info("🚀 MedEdge API starting up...")

    for path in [MODEL_PATH, SCALER_PATH, LE_DIAG_PATH, LE_MED_PATH]:
        if not os.path.exists(path):
            raise RuntimeError(
                f"Required file '{path}' not found. "
                "Run train_mededge_v4.py first."
            )

    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"   Device: {state.device}")

    # Load support objects
    state.scaler  = joblib.load(SCALER_PATH)
    state.le_diag = joblib.load(LE_DIAG_PATH)
    state.le_med  = joblib.load(LE_MED_PATH)
    logger.info("   ✅ Scaler and encoders loaded")

    # Load model
    checkpoint  = torch.load(MODEL_PATH, map_location=state.device)
    state.cfg   = checkpoint["config"]
    state.model = MedEdgeV4(
        len(state.le_diag.classes_),
        len(state.le_med.classes_),
        len(VITALS_COLS),
        state.cfg,
    ).to(state.device)
    state.model.load_state_dict(checkpoint["model_state"])
    state.model.eval()   # ← disables dropout for inference
    logger.info(f"   ✅ Model loaded (best epoch: {checkpoint.get('epoch', '?')}  "
                f"val_auc: {checkpoint.get('val_auc', 0):.4f})")

    logger.info("✅ MedEdge API is ready.\n")
    yield

    # ── SHUTDOWN ──────────────────────────────────────────────
    logger.info("👋 MedEdge API shutting down.")


# ══════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════
app = FastAPI(
    title="MedEdge Risk API",
    description="Predicts 30-day hospital readmission risk.",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock this down in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════
class VitalsInput(BaseModel):
    Blood_Glucose           : float = Field(..., ge=0,   le=1000, example=110.0)
    Creatinine              : float = Field(..., ge=0,   le=30,   example=1.1)
    Hemoglobin              : float = Field(..., ge=0,   le=25,   example=13.5)
    WBC                     : float = Field(..., ge=0,   le=100,  example=7.2)
    Heart_Rate              : float = Field(..., ge=20,  le=300,  example=82.0)
    Blood_Pressure_Systolic : float = Field(..., ge=50,  le=300,  example=130.0)
    SpO2                    : float = Field(..., ge=50,  le=100,  example=97.0)
    Temperature             : float = Field(..., ge=90,  le=115,  example=98.6)


class PredictRequest(BaseModel):
    diagnosis  : str        = Field(..., example="Heart Failure")
    medication : str        = Field(..., example="Metoprolol")
    vitals     : VitalsInput
    threshold  : Optional[float] = Field(RISK_THRESHOLD, ge=0.01, le=0.99)

    @validator("diagnosis", "medication")
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Cannot be empty")
        return v.strip()


class PredictResponse(BaseModel):
    prediction   : str    # "High Risk" | "Low Risk"
    risk_score   : float  # 0.0 – 1.0
    probability  : str    # "73.2%"
    threshold    : float
    model_version: str


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "MedEdge Risk API v4"}


@app.get("/health", tags=["health"])
def health():
    """Liveness probe — returns 200 if the model is loaded."""
    if state.model is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "status"       : "healthy",
        "device"       : str(state.device),
        "num_diagnoses": len(state.le_diag.classes_),
        "num_meds"     : len(state.le_med.classes_),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # ── Encode text safely (Unknowns become 0) ─────────────────
    diag_idx = int(state.le_diag.transform([req.diagnosis])[0])
    med_idx  = int(state.le_med.transform([req.medication])[0])

    # ── Scale vitals ──────────────────────────────────────────
    vitals_array = np.array(
        [[getattr(req.vitals, c) for c in VITALS_COLS]],
        dtype=np.float32
    )
    vitals_scaled = state.scaler.transform(vitals_array)

    # ── Inference ─────────────────────────────────────────────
    with torch.no_grad():
        logit = state.model(
            torch.tensor([diag_idx],     dtype=torch.long).to(state.device),
            torch.tensor([med_idx],      dtype=torch.long).to(state.device),
            torch.tensor(vitals_scaled,  dtype=torch.float32).to(state.device),
        )
        prob = torch.sigmoid(logit).item()

    return PredictResponse(
        prediction    = "High Risk" if prob >= req.threshold else "Low Risk",
        risk_score    = round(prob * 100, 2),
        probability   = f"{prob:.1%}",
        threshold     = req.threshold,
        model_version = "MedEdgeV4",
    )

@app.get("/classes/diagnoses", tags=["metadata"])
def list_diagnoses():
    """Returns all diagnosis labels the model was trained on."""
    return {"diagnoses": sorted(state.le_diag.classes_.tolist())}


@app.get("/classes/medications", tags=["metadata"])
def list_medications():
    """Returns all medication labels the model was trained on."""
    return {"medications": sorted(state.le_med.classes_.tolist())}