import os
from pathlib import Path

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split


from ml.models.viability_model import ViabilityModel
from ml.models.price_model import ConversionModel
from ml.models.stockout_model import StockoutRiskModel
from ml.models.clustering_model import ClusteringModel  # ← NEW


# CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "product_intelligence_supplier_enhanced.xlsx"

MODEL_ROOT = PROJECT_ROOT / "data" / "models"

os.makedirs(MODEL_ROOT / "viability", exist_ok=True)
os.makedirs(MODEL_ROOT / "price_optimizer", exist_ok=True)
os.makedirs(MODEL_ROOT / "stockout_risk", exist_ok=True)
os.makedirs(MODEL_ROOT / "clustering", exist_ok=True)

print("📥 Loading dataset...")
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Training data not found at {INPUT_FILE}. "
        "Place your Excel file there or update INPUT_FILE in ml/train/train_models.py."
    )

df = pd.read_excel(INPUT_FILE)
print("Loaded:", df.shape)


df["landed_cost"] = df["cost"] + df["shipping_cost"] + df["duties"]
df["margin"] = (df["price"] - df["landed_cost"]) / df["price"]
df["margin"] = df["margin"].clip(lower=0)

df["sale_30d"] = ((df["margin"] > 0.10) & (df["inventory"] > 10)).astype(int)
df["conversion_flag"] = (df["margin"] > 0.08).astype(int)
df["stockout_flag"] = ((df["stock"] < 25) | (df["lead_time_days"] > 14)).astype(int)


FEATURES = [
    "price", "cost", "shipping_cost", "duties",
    "lead_time_days", "stock", "inventory", "quantity",
    "demand", "past_sales",
    "weight_kg", "length_cm", "width_cm", "height_cm",
    "margin", "supplier_reliability_score"
]

X = df[FEATURES].fillna(0)

y_viability = df["sale_30d"]
y_conversion = df["conversion_flag"]
y_stockout = df["stockout_flag"]


print("🔵 Training ViabilityModel...")
viability_model = ViabilityModel()
viability_model.train(X, y_viability)

viability_model.save(str(MODEL_ROOT / "viability" / "model.pkl"))
print("✅ Saved viability model")


print("🟢 Training ConversionModel...")
conv_model = ConversionModel()
conv_model.train(X, y_conversion)

conv_model.save(str(MODEL_ROOT / "price_optimizer" / "conversion_model.pkl"))
print("✅ Saved conversion model")


print("🟠 Training StockoutRiskModel...")
stockout_model = StockoutRiskModel()
stockout_model.train(X, y_stockout)

stockout_model.save(str(MODEL_ROOT / "stockout_risk" / "model.pkl"))
print("✅ Saved stockout risk model")


print("🟣 Training ClusteringModel (TF-IDF + KMeans text clustering)...")

def build_text(row):
    parts = [
        str(row.get("product_name", "")),
        str(row.get("description", "")),
        str(row.get("category", "")),
    ]
    return " ".join([p for p in parts if p.strip()])

product_texts = df.apply(build_text, axis=1).tolist()

cluster_model = ClusteringModel(config={"n_clusters": 6})
cluster_model.train(product_texts)

cluster_model.save(str(MODEL_ROOT / "clustering" / "model.pkl"))
print("✅ Saved clustering model (TF-IDF + KMeans)")


print("\n🎉 ALL MODELS TRAINED & SAVED SUCCESSFULLY!")
print("📁 Output directory:", MODEL_ROOT)
