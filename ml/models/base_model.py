from abc import ABC, abstractmethod
import joblib
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


class BaseModel(ABC):
   

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.version = self.config.get("version", "1.0.0")
        self.metadata: Dict[str, Any] = {}

 
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
       
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        
        pass

    def predict_proba(self, X: pd.DataFrame) -> Any:
       
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_proba()."
        )


    def save(self, filepath: str) -> None:
        
        filepath = str(filepath)

        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the core ML model
        joblib.dump(self.model, filepath, protocol=4)

       
        meta_path = filepath.replace(".pkl", "_meta.json")
        self.metadata = {
            "version": self.version,
            "config": self.config,
            "saved_at": datetime.now().isoformat(),
            "class_name": self.__class__.__name__,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4)

        print(f"✔ Saved model → {filepath}")

    def load(self, filepath: str) -> None:
        
        filepath = str(filepath)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")

        self.model = joblib.load(filepath)
        self.is_trained = True

        meta_path = filepath.replace(".pkl", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                self.metadata = json.load(f)

        print(f"✔ Loaded model → {filepath}")
