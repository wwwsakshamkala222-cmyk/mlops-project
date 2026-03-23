from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

# 1. THE BRAIN STRUCTURE
class MedEdgeBeast(nn.Module):
    def __init__(self, vocab_diag, vocab_med):
        super(MedEdgeBeast, self).__init__()
        self.diag_emb = nn.Embedding(vocab_diag, 32)
        self.med_emb = nn.Embedding(vocab_med, 32)
        self.core = nn.Sequential(
            nn.Linear(32 + 32 + 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, diag, med, bio):
        d = self.diag_emb(diag); m = self.med_emb(med)
        x = torch.cat((d, m, bio), dim=1)
        return self.core(x)

app = FastAPI(title="BEAST v2.0 API")

# ✅ THE FIX: CORS MIDDLEWARE (Allows your HTML to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD ASSETS (Explicitly check if they exist)
try:
    # We use os.getcwd() to make sure we are looking in the right folder
    base_path = os.getcwd()
    le_diag = joblib.load(os.path.join(base_path, "le_diag.pkl"))
    le_med = joblib.load(os.path.join(base_path, "le_med.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    
    model = MedEdgeBeast(len(le_diag.classes_), len(le_med.classes_))
    model.load_state_dict(torch.load("mededge_beast_100k.pth", map_location='cpu'))
    model.eval()
    print("✅ Clinical Brain & Assets Loaded Successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load assets: {e}")

# 3. DATA FORMAT (Matches your app.js vitals list)
class PatientRequest(BaseModel):
    diagnosis: str
    medication: str
    vitals: list # [spo2, hr, gluc, sbp, dbp, temp, rr, crea]

@app.post("/predict")
async def predict(request: PatientRequest):
    try:
        d_idx = le_diag.transform([request.diagnosis])[0]
        m_idx = le_med.transform([request.medication])[0]
        
        # 🚨 THE FIX: No more scrambling! 
        # The website array perfectly matches the model's memory.
        v_scaled = scaler.transform(np.array(request.vitals).reshape(1, -1))
        
        d_t = torch.tensor([d_idx], dtype=torch.long)
        m_t = torch.tensor([m_idx], dtype=torch.long)
        b_t = torch.tensor(v_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            prob = torch.sigmoid(model(d_t, m_t, b_t)).item()
            
        return {
            "risk_score": prob, # App.js will convert this to a percentage
            "feature_importance": None # Handled by app.js for now
        }
    except Exception as e:
        print(f"❌ PREDICTION ERROR: {e}")
        raise HTTPException(status_code=400, detail=str(e))