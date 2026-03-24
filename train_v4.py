"""
╔══════════════════════════════════════════════════════════════╗
║           MedEdge Risk Model — Training Pipeline V4          ║
║         Hospital Readmission Prediction · PyTorch            ║
╚══════════════════════════════════════════════════════════════╝

WHAT'S NEW vs V3:
  ✅ pos_weight in loss   → fixes silent class-imbalance failure
  ✅ Mini-batch DataLoader → real SGD, better generalisation
  ✅ LR scheduler          → ReduceLROnPlateau (adaptive)
  ✅ Early stopping        → saves best checkpoint, not last
  ✅ AUC-ROC + F1 eval     → metrics that matter for medical models
  ✅ Residual connections  → deeper gradient flow without vanishing
  ✅ Multi-head architecture → separate pathways per modality
  ✅ Stratified split       → preserves class ratio in train/val/test
  ✅ Config block           → all hyperparams in one place
  ✅ Reproducibility seed   → same results every run
  ✅ Inference helper       → ready-to-use predict() function
"""

import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix
)
import joblib

# ══════════════════════════════════════════════════════════════
#  CONFIG — change everything here, touch nothing else
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
#  CONFIG — change everything here, touch nothing else
# ══════════════════════════════════════════════════════════════
CONFIG = {
    # ── Data ──────────────────────────────────────────────────
    "csv_name"        : "mimic_iii_data.csv",   
    "diagnosis_col"   : "Diagnoses",             # ✅ FIXED
    "medication_col"  : "Medications",           # ✅ FIXED
    "target_col"      : "Readmission_Flag",      # ✅ FIXED
    "vitals_cols"     : [                        # These were already correct!
        "Blood_Glucose", "Creatinine", "Hemoglobin", "WBC",
        "Heart_Rate", "Blood_Pressure_Systolic", "SpO2", "Temperature"
    ],

    # ── Split ─────────────────────────────────────────────────
    "test_size"       : 0.15,
    "val_size"        : 0.15,
    "random_seed"     : 42,

    # ── Model ─────────────────────────────────────────────────
    "diag_emb_dim"    : 16,
    "med_emb_dim"     : 16,
    "hidden_dims"     : [128, 64, 32],
    "dropout"         : 0.35,

    # ── Training ──────────────────────────────────────────────
    "epochs"          : 100,
    "batch_size"      : 128,
    "lr"              : 5e-4,
    "weight_decay"    : 1e-4,
    "lr_patience"     : 8,
    "lr_factor"       : 0.5,
    "early_stop_patience" : 20,

    # ── Output ────────────────────────────────────────────────
    "model_path"      : "mededge_v4_best.pth",
    "scaler_path"     : "scaler.pkl",
    "le_diag_path"    : "le_diag.pkl",
    "le_med_path"     : "le_med.pkl",
}

# ══════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG["random_seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
#  MODEL — Multi-head with residual connections
# ══════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    """
    Two-layer residual block with BatchNorm + Dropout.
    If in_dim != out_dim, a 1×1 projection aligns the skip connection.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc1   = nn.Linear(in_dim, out_dim)
        self.bn1   = nn.BatchNorm1d(out_dim)
        self.fc2   = nn.Linear(out_dim, out_dim)
        self.bn2   = nn.BatchNorm1d(out_dim)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = torch.relu(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        return torch.relu(out + residual)   # ← residual addition


class MedEdgeV4(nn.Module):
    """
    Multi-modal architecture:
      • Diagnosis  → embedding → private 32-d projection
      • Medication → embedding → private 32-d projection
      • Vitals     → private 32-d MLP
      • All three concatenated → shared residual trunk → binary output
    """
    def __init__(self, num_diags: int, num_meds: int,
                 num_vitals: int, cfg: dict):
        super().__init__()
        ed = cfg["diag_emb_dim"]
        em = cfg["med_emb_dim"]
        dr = cfg["dropout"]
        hd = cfg["hidden_dims"]   # e.g. [128, 64, 32]

        # ── Per-modality heads ────────────────────────────────
        self.diag_emb  = nn.Embedding(num_diags, ed, padding_idx=0)
        self.med_emb   = nn.Embedding(num_meds,  em, padding_idx=0)

        self.diag_head = nn.Sequential(
            nn.Linear(ed, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dr)
        )
        self.med_head  = nn.Sequential(
            nn.Linear(em, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dr)
        )
        self.vital_head = nn.Sequential(
            nn.Linear(num_vitals, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dr)
        )

        # ── Shared residual trunk ─────────────────────────────
        trunk_in = 32 + 32 + 32   # 96
        layers = []
        prev = trunk_in
        for h in hd:
            layers.append(ResidualBlock(prev, h, dr))
            prev = h
        self.trunk = nn.Sequential(*layers)

        # ── Output ────────────────────────────────────────────
        self.out = nn.Linear(prev, 1)

        # ── Weight init ───────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, diag, med, vitals):
        d = self.diag_head(self.diag_emb(diag))
        m = self.med_head(self.med_emb(med))
        v = self.vital_head(vitals)
        x = torch.cat([d, m, v], dim=1)
        x = self.trunk(x)
        return self.out(x)   # raw logits — sigmoid applied at eval time


# ══════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════

print("\n📂 Loading data...")
df = pd.read_csv(CONFIG["csv_name"])
print(f"   Rows: {len(df):,}  |  Cols: {df.shape[1]}")

# ── THE NEW SAFE ENCODER ──────────────────────────────────────
class SafeLabelEncoder:
    def __init__(self):
        self.classes_ = np.array([])
        self.class_to_idx = {}

    def fit_transform(self, y):
        # Add <UNKNOWN> at index 0
        unique_classes = np.unique(y)
        self.classes_ = np.insert(unique_classes, 0, "<UNKNOWN>")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        return np.array([self.class_to_idx.get(x, 0) for x in y])

    def transform(self, y):
        # If it doesn't know the word, it safely returns 0 (<UNKNOWN>)
        return np.array([self.class_to_idx.get(x, 0) for x in y])

le_diag = SafeLabelEncoder()
le_med  = SafeLabelEncoder()

df["diagnosis_encoded"] = le_diag.fit_transform(df[CONFIG["diagnosis_col"]].astype(str))
df["medication_encoded"] = le_med.fit_transform(df[CONFIG["medication_col"]].astype(str))

# ── Features & target ─────────────────────────────────────────
vitals_cols = CONFIG["vitals_cols"]
X_vitals    = df[vitals_cols].values.astype(np.float32)
X_diag      = df["diagnosis_encoded"].values
X_med       = df["medication_encoded"].values
y           = df[CONFIG["target_col"]].values.astype(np.float32)

pos_rate = y.mean()
print(f"   Positive rate (readmitted): {pos_rate:.1%}")

# ── Stratified 70 / 15 / 15 split ─────────────────────────────
(X_v_tv, X_v_test,
 X_d_tv, X_d_test,
 X_m_tv, X_m_test,
 y_tv,   y_test) = train_test_split(
    X_vitals, X_diag, X_med, y,
    test_size=CONFIG["test_size"],
    stratify=y,
    random_state=CONFIG["random_seed"]
)

val_fraction = CONFIG["val_size"] / (1 - CONFIG["test_size"])
(X_v_train, X_v_val,
 X_d_train, X_d_val,
 X_m_train, X_m_val,
 y_train,   y_val) = train_test_split(
    X_v_tv, X_d_tv, X_m_tv, y_tv,
    test_size=val_fraction,
    stratify=y_tv,
    random_state=CONFIG["random_seed"]
)

print(f"   Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

# ── Scale vitals (fit ONLY on train) ──────────────────────────
scaler = StandardScaler()
X_v_train = scaler.fit_transform(X_v_train)
X_v_val   = scaler.transform(X_v_val)
X_v_test  = scaler.transform(X_v_test)

# ── Build DataLoaders ─────────────────────────────────────────
def make_loader(v, d, m, labels, shuffle=False):
    ds = TensorDataset(
        torch.tensor(d, dtype=torch.long),
        torch.tensor(m, dtype=torch.long),
        torch.tensor(v, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32).view(-1, 1),
    )
    return DataLoader(ds, batch_size=CONFIG["batch_size"],
                      shuffle=shuffle, num_workers=0, pin_memory=False)

train_loader = make_loader(X_v_train, X_d_train, X_m_train, y_train, shuffle=True)
val_loader   = make_loader(X_v_val,   X_d_val,   X_m_val,   y_val)
test_loader  = make_loader(X_v_test,  X_d_test,  X_m_test,  y_test)


# ══════════════════════════════════════════════════════════════
#  BUILD MODEL
# ══════════════════════════════════════════════════════════════
num_diags  = len(le_diag.classes_)
num_meds   = len(le_med.classes_)
num_vitals = len(vitals_cols)

model = MedEdgeV4(num_diags, num_meds, num_vitals, CONFIG).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🧠 Model V4 | Trainable parameters: {total_params:,}")

# ── Loss: pos_weight corrects for class imbalance ─────────────
neg_count  = (y_train == 0).sum()
pos_count  = (y_train == 1).sum()
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(DEVICE)
print(f"   pos_weight = {pos_weight.item():.2f}  "
      f"(neg={neg_count:,}  pos={pos_count:,})")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"]
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=CONFIG["lr_patience"],
    factor=CONFIG["lr_factor"],
)


# ══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════
def run_epoch(loader, train=True):
    model.train(train)
    total_loss, all_probs, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for d_b, m_b, v_b, y_b in loader:
            d_b, m_b, v_b, y_b = (
                d_b.to(DEVICE), m_b.to(DEVICE),
                v_b.to(DEVICE), y_b.to(DEVICE)
            )

            if train:
                optimizer.zero_grad()

            logits = model(d_b, m_b, v_b)
            loss   = criterion(logits, y_b)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_b)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y_b.cpu().numpy().flatten())

    n        = len(all_labels)
    avg_loss = total_loss / n
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return avg_loss, auc


print(f"\n🚀 Training for up to {CONFIG['epochs']} epochs...\n")
print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Train AUC':>10}  "
      f"{'Val Loss':>9}  {'Val AUC':>8}  {'LR':>9}")
print("─" * 68)

best_val_loss    = float("inf")
patience_counter = 0
best_epoch       = 0

for epoch in range(1, CONFIG["epochs"] + 1):
    tr_loss, tr_auc = run_epoch(train_loader, train=True)
    vl_loss, vl_auc = run_epoch(val_loader,   train=False)
    scheduler.step(vl_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"{epoch:>6}  {tr_loss:>11.4f}  {tr_auc:>10.4f}  "
          f"{vl_loss:>9.4f}  {vl_auc:>8.4f}  {current_lr:>9.2e}")

    # ── Checkpoint best model ──────────────────────────────────
    if vl_loss < best_val_loss:
        best_val_loss    = vl_loss
        best_epoch       = epoch
        patience_counter = 0
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "val_loss"   : vl_loss,
            "val_auc"    : vl_auc,
            "config"     : CONFIG,
        }, CONFIG["model_path"])
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["early_stop_patience"]:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch} "
                  f"(best was epoch {best_epoch})")
            break

print(f"\n✅ Best checkpoint: epoch {best_epoch}  |  val_loss={best_val_loss:.4f}")


# ══════════════════════════════════════════════════════════════
#  FINAL EVALUATION ON TEST SET
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 68)
print("  FINAL TEST SET EVALUATION (best checkpoint)")
print("═" * 68)

# Load best weights
checkpoint = torch.load(CONFIG["model_path"], map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state"])
model.eval()

all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for d_b, m_b, v_b, y_b in test_loader:
        logits = model(d_b.to(DEVICE), m_b.to(DEVICE), v_b.to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        preds  = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y_b.numpy().flatten())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

auc_roc = roc_auc_score(all_labels, all_probs)
auc_pr  = average_precision_score(all_labels, all_probs)   # better for imbalanced data
cm      = confusion_matrix(all_labels, all_preds)

print(f"\n  AUC-ROC  : {auc_roc:.4f}  (>0.80 = good, >0.90 = great)")
print(f"  AUC-PR   : {auc_pr:.4f}   (precision-recall; better metric for imbalanced data)")
print(f"\n  Confusion Matrix:")
print(f"               Pred 0   Pred 1")
print(f"  Actual 0   {cm[0,0]:>7,}  {cm[0,1]:>7,}")
print(f"  Actual 1   {cm[1,0]:>7,}  {cm[1,1]:>7,}")
print()
print(classification_report(all_labels, all_preds,
                             target_names=["Not Readmitted", "Readmitted"]))


# ══════════════════════════════════════════════════════════════
#  SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════
joblib.dump(scaler, CONFIG["scaler_path"])
joblib.dump(le_diag, CONFIG["le_diag_path"])
joblib.dump(le_med,  CONFIG["le_med_path"])

print("💾 Saved:")
print(f"   • {CONFIG['model_path']}   ← best model weights + metadata")
print(f"   • {CONFIG['scaler_path']}  ← vitals scaler")
print(f"   • {CONFIG['le_diag_path']} ← diagnosis encoder")
print(f"   • {CONFIG['le_med_path']}  ← medication encoder")


# ══════════════════════════════════════════════════════════════
#  INFERENCE HELPER  (import this in your app)
# ══════════════════════════════════════════════════════════════
def predict(
    diagnosis: str,
    medication: str,
    vitals_dict: dict,
    threshold: float = 0.5
) -> dict:
    """
    Predict readmission risk for a single patient.

    Args:
        diagnosis   : e.g. "Heart Failure"
        medication  : e.g. "Metoprolol"
        vitals_dict : {
            "Blood_Glucose": 120, "Creatinine": 1.1, "Hemoglobin": 13.5,
            "WBC": 7.2, "Heart_Rate": 82, "Blood_Pressure_Systolic": 130,
            "SpO2": 97, "Temperature": 98.6
        }
        threshold   : probability cutoff (default 0.5)

    Returns:
        {
          "risk_score"  : float (0–1),
          "prediction"  : "High Risk" | "Low Risk",
          "probability" : float
        }
    """
    _scaler  = joblib.load(CONFIG["scaler_path"])
    _le_diag = joblib.load(CONFIG["le_diag_path"])
    _le_med  = joblib.load(CONFIG["le_med_path"])

    ckpt   = torch.load(CONFIG["model_path"], map_location="cpu")
    _model = MedEdgeV4(
        len(_le_diag.classes_), len(_le_med.classes_),
        len(CONFIG["vitals_cols"]), ckpt["config"]
    )
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()

    # Encode inputs
    diag_idx = _le_diag.transform([diagnosis])[0]
    med_idx  = _le_med.transform([medication])[0]
    vitals   = np.array([[vitals_dict[c] for c in CONFIG["vitals_cols"]]], dtype=np.float32)
    vitals   = _scaler.transform(vitals)

    with torch.no_grad():
        logit = _model(
            torch.tensor([diag_idx], dtype=torch.long),
            torch.tensor([med_idx],  dtype=torch.long),
            torch.tensor(vitals,     dtype=torch.float32),
        )
        prob = torch.sigmoid(logit).item()

    return {
        "risk_score" : round(prob, 4),
        "prediction" : "High Risk" if prob >= threshold else "Low Risk",
        "probability": f"{prob:.1%}",
    }


# ── Quick sanity-check inference example ──────────────────────
print("\n" + "═" * 68)
print("  SAMPLE INFERENCE")
print("═" * 68)
sample_diag = le_diag.classes_[0]
sample_med  = le_med.classes_[0]
sample_vitals = {c: float(df[c].median()) for c in vitals_cols}

result = predict(sample_diag, sample_med, sample_vitals)
print(f"  Patient: diag='{sample_diag}', med='{sample_med}'")
print(f"  Result : {result}")
print("\n🏁 All done — MedEdge V4 is trained and ready.\n")