import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from ml.models.base_model import BaseModel


class ConversionModel(BaseModel):
    

    FEATURES = [
        "price", "cost", "shipping_cost", "duties",
        "lead_time_days", "stock", "inventory", "quantity",
        "demand", "past_sales",
        "weight_kg", "length_cm", "width_cm", "height_cm",
        "margin", "supplier_reliability_score"
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self.model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs"
        )
        self.is_trained = False

 
    def train(self, X: pd.DataFrame, y: pd.Series):
        X = X[self.FEATURES].fillna(0)

        unique_classes = np.unique(y)
        if unique_classes.size < 2:
            constant_class = int(unique_classes[0])
            self.model = DummyClassifier(strategy="constant", constant=constant_class)
            self.model.fit(X, y)
            self.is_trained = True
            print(
                f"⚠ ConversionModel trained with DummyClassifier "
                f"(only one class present: {constant_class})"
            )
            return

        self.model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs"
        )
        self.model.fit(X, y)
        self.is_trained = True


    def predict(self, X: pd.DataFrame):
        
        X = X[self.FEATURES].fillna(0)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        
        X = X[self.FEATURES].fillna(0)
        proba = self.model.predict_proba(X)

        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return np.zeros(len(X))

        idx = np.where(classes == 1)[0]
        if idx.size == 0:
            return np.zeros(len(X))

        return proba[:, idx[0]]

    def predict_conversion_probability(self, product: dict, price: float) -> float:
       
        row = {feat: product.get(feat, 0) for feat in self.FEATURES}
        row["price"] = price
        row["margin"] = (
            (price - (product.get("cost", 0)
                      + product.get("shipping_cost", 0)
                      + product.get("duties", 0)))
            / price
        )

        X = pd.DataFrame([row])
        return float(self.predict_proba(X)[0])

    def predict_for_price(self, product: dict, price: float) -> float:
        """Alias used inside PriceOptimizer — clean, no kwargs."""
        return self.predict_conversion_probability(product, price)

 
    def save(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "config": self.config,
                    "is_trained": self.is_trained,
                },
                f,
            )

    def load(self, filepath):
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.config = data.get("config", {})
        self.is_trained = data.get("is_trained", True)
