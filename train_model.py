"""
============================================================
  Orange Telecom Churn Prediction - Model Training Script
  Author  : Senior ML Engineer
  Purpose : Train a LightGBM pipeline and persist to disk
============================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = "Orange_Telecom_Churn_Data.csv"
MODEL_PATH  = "model.pkl"
TARGET_COL  = "churned"
DROP_COLS   = ["phone_number"]          # unique identifier – no signal
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# 2. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise target to int
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    print(f"[DATA]  Loaded {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"[DATA]  Churn rate: {df[TARGET_COL].mean()*100:.1f}%")
    return df

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ratio features
    df["day_charge_per_min"]   = df["total_day_charge"]   / (df["total_day_minutes"]   + 1e-6)
    df["eve_charge_per_min"]   = df["total_eve_charge"]   / (df["total_eve_minutes"]   + 1e-6)
    df["night_charge_per_min"] = df["total_night_charge"] / (df["total_night_minutes"] + 1e-6)
    df["intl_charge_per_min"]  = df["total_intl_charge"]  / (df["total_intl_minutes"]  + 1e-6)

    df["total_minutes"]  = (df["total_day_minutes"] + df["total_eve_minutes"] +
                            df["total_night_minutes"] + df["total_intl_minutes"])
    df["total_calls"]    = (df["total_day_calls"] + df["total_eve_calls"] +
                            df["total_night_calls"] + df["total_intl_calls"])
    df["total_charge"]   = (df["total_day_charge"] + df["total_eve_charge"] +
                            df["total_night_charge"] + df["total_intl_charge"])

    df["charge_per_call"] = df["total_charge"] / (df["total_calls"] + 1e-6)
    df["charge_per_day"]  = df["total_charge"] / (df["account_length"] + 1e-6)
    df["calls_per_day"]   = df["total_calls"]  / (df["account_length"] + 1e-6)

    # Service call ratio
    df["service_call_rate"] = (df["number_customer_service_calls"] /
                               (df["account_length"] + 1e-6))

    print(f"[FEAT]  After engineering: {df.shape[1]} columns")
    return df

# ─────────────────────────────────────────────
# 4. BUILD SKLEARN PIPELINE
# ─────────────────────────────────────────────
def build_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")

    # LightGBM with class-imbalance handling via scale_pos_weight
    lgbm = LGBMClassifier(
        n_estimators      = 800,
        learning_rate     = 0.05,
        max_depth         = 6,
        num_leaves        = 50,
        min_child_samples = 20,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        scale_pos_weight  = 6,   # ~85:15 class ratio -> boost minority
        random_state      = RANDOM_SEED,
        verbosity         = -1,
        n_jobs            = -1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf",          lgbm),
    ])

# ─────────────────────────────────────────────
# 5. TRAIN & EVALUATE
# ─────────────────────────────────────────────
def train(df: pd.DataFrame):
    df = engineer_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"[FEAT]  Numeric: {len(num_cols)}  Categorical: {len(cat_cols)}")

    pipeline = build_pipeline(num_cols, cat_cols)

    # Cross-validation AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"[EVAL]  CV AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    # Final fit on all data
    pipeline.fit(X, y)

    # Report feature importances
    lgbm_model   = pipeline.named_steps["clf"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feat_names   = num_cols + cat_cols
    importances  = lgbm_model.feature_importances_
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi_df.sort_values("importance", ascending=False, inplace=True)
    print("\n[FEAT]  Top-15 important features:")
    print(fi_df.head(15).to_string(index=False))

    return pipeline, auc_scores.mean(), num_cols, cat_cols

# ─────────────────────────────────────────────
# 6. SAVE ARTEFACTS
# ─────────────────────────────────────────────
def save_artefacts(pipeline, auc: float, num_cols: list, cat_cols: list,
                   df_raw: pd.DataFrame):
    artefacts = {
        "pipeline"  : pipeline,
        "auc"       : round(auc, 4),
        "num_cols"  : num_cols,
        "cat_cols"  : cat_cols,
        # Store distribution stats for random customer generation
        "data_stats": compute_stats(df_raw),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artefacts, f)
    print(f"\n[SAVE]  Model saved to {MODEL_PATH}  (AUC={auc:.4f})")

def compute_stats(df: pd.DataFrame) -> dict:
    """Store per-column distributions so the app can generate realistic customers."""
    stats = {}
    for col in df.columns:
        if col == TARGET_COL:
            continue
        s = df[col]
        if s.dtype in [np.float64, np.int64, np.float32, np.int32]:
            stats[col] = {
                "type": "numeric",
                "min":  float(s.min()),
                "max":  float(s.max()),
                "mean": float(s.mean()),
                "std":  float(s.std()),
                "p25":  float(s.quantile(0.25)),
                "p75":  float(s.quantile(0.75)),
            }
        else:
            stats[col] = {
                "type":   "categorical",
                "values": s.dropna().unique().tolist(),
                "freq":   (s.value_counts(normalize=True).to_dict()),
            }
    return stats

# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Orange Telecom Churn Model Training")
    print("=" * 55)
    df_raw = load_data(DATA_PATH)
    pipeline, auc, num_cols, cat_cols = train(df_raw.copy())
    save_artefacts(pipeline, auc, num_cols, cat_cols, df_raw)
    print("\n✅  Training complete!")
