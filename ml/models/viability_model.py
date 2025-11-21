"""
Viability Model — Predicts probability of sale within 30 days
Course-Aligned (RandomForest + SHAP)
"""

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from ml.models.base_model import BaseModel


class ViabilityModel(BaseModel):
    """
    Predicts P(sale within 30 days).
    Course-aligned:
    - RandomForestClassifier
    - SHAP TreeExplainer
    """

    def __init__(self, config=None):
        super().__init__(config)

        self.model = RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            class_weight="balanced",
            random_state=42
        )

        self.explainer = None
        self.feature_names = []

    # ============================================================
    # TRAIN MODEL
    # ============================================================
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)

        # Fit model
        self.model.fit(X, y)
        self.is_trained = True

        # SHAP TreeExplainer (works perfectly with RandomForest)
        self.explainer = shap.TreeExplainer(self.model)

        # Handle case where only a single class is present in y
        unique_classes = np.unique(y)
        proba = self.model.predict_proba(X)
        if proba.shape[1] >= 2:
            preds = proba[:, 1]
        else:
            # Single-column probabilities; use that column directly
            preds = proba[:, 0]

        # AUC requires at least two classes; avoid calling it when only one class
        if unique_classes.size < 2:
            print(
                "✔ Viability model trained "
                "(AUC not computed – only one class present in labels)"
            )
        else:
            auc = roc_auc_score(y, preds)
            print(f"✔ Viability model trained (AUC = {auc:.4f})")

    # ============================================================
    # PREDICT LABEL
    # ============================================================
    def predict(self, X: pd.DataFrame):
        self._check_features(X)
        return self.model.predict(X)

    # ============================================================
    # PREDICT PROBABILITY
    # ============================================================
    def predict_proba(self, X: pd.DataFrame):
        self._check_features(X.copy())
        return self.model.predict_proba(X)[:, 1]
    
    # ============================================================
    # PREDICT VIABILITY SCORE (alias for predict_proba)
    # ============================================================
    def predict_viability_score(self, X: pd.DataFrame):
        """Predict viability score (probability of sale within 30 days)"""
        return self.predict_proba(X)

    # ============================================================
    # EXPLAINABILITY (SHAP)
    # ============================================================
    def explain(self, X: pd.DataFrame):
        self._check_features(X)

        # SHAP v0.44+ returns NUMPY ARRAY for RF
        shap_values = self.explainer.shap_values(X)

        base_value = self.explainer.expected_value

        feature_importance = np.abs(shap_values).mean(axis=0)

        per_sample = []
        for i in range(len(X)):
            per_sample.append({
                "prediction": float(self.predict_proba(X.iloc[[i]])[0]),
                "base_value": float(base_value),
                "feature_contributions": {
                    feature: float(shap_values[i][j])
                    for j, feature in enumerate(self.feature_names)
                }
            })

        return {
            "base_value": float(base_value),
            "feature_importance": {
                self.feature_names[i]: float(feature_importance[i])
                for i in range(len(self.feature_names))
            },
            "per_sample_explanations": per_sample,
            "shap_values": shap_values.tolist()
        }

    # ============================================================
    # INTERNAL CHECK
    # ============================================================
    def _check_features(self, X: pd.DataFrame):
        missing = set(self.feature_names) - set(X.columns)
        extra = set(X.columns) - set(self.feature_names)
        
        for c in missing:
            X[c] = 0.0
    
        X = X[self.feature_names]
