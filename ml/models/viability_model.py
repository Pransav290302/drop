import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from ml.models.base_model import BaseModel


class ViabilityModel(BaseModel):
   
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

 
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)

        # Fit model
        self.model.fit(X, y)
        self.is_trained = True

      
        self.explainer = shap.TreeExplainer(self.model)

      # Handle case where only a single class is present in y
        unique_classes = np.unique(y)
        proba = self.model.predict_proba(X)
        if proba.shape[1] >= 2:
            preds = proba[:, 1]
        else:
           
            preds = proba[:, 0]

      
        if unique_classes.size < 2:
            print(
                "✔ Viability model trained "
                "(AUC not computed – only one class present in labels)"
            )
        else:
            auc = roc_auc_score(y, preds)
            print(f"✔ Viability model trained (AUC = {auc:.4f})")

  
    def predict(self, X: pd.DataFrame):
        self._check_features(X)
        return self.model.predict(X)


    def predict_proba(self, X: pd.DataFrame):
        self._check_features(X.copy())
        return self.model.predict_proba(X)[:, 1]
    

    def predict_viability_score(self, X: pd.DataFrame):
        
        return self.predict_proba(X)


    def explain(self, X: pd.DataFrame):
        self._check_features(X)

       
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


    def _check_features(self, X: pd.DataFrame):
        missing = set(self.feature_names) - set(X.columns)
        extra = set(X.columns) - set(self.feature_names)
        
        for c in missing:
            X[c] = 0.0
    
        X = X[self.feature_names]
