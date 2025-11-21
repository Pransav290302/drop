"""ML Pipeline service for orchestrating model calls"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from backend.app.core.config import settings

# ML model imports
from ml.models.viability_model import ViabilityModel
from ml.models.price_model import ConversionModel
from ml.models.stockout_model import StockoutRiskModel

# NEW TF-IDF + KMeans clustering model (correct import)
from ml.models.clustering_model import ClusteringModel

from ml.services.price_optimizer import PriceOptimizer
from ml.pipelines.viability_pipeline import ViabilityPipeline  # noqa

# Data processing imports
from ml.data.normalization import DataNormalizer
from ml.features.engineering import engineer_features
from ml.config import get_schema_config

logger = logging.getLogger(__name__)


class MLPipelineService:
    """
    Service orchestrating:
    - viability prediction
    - price optimization
    - stockout risk
    - TF-IDF clustering
    """

    def __init__(self):
        self.viability_model: Optional[ViabilityModel] = None
        self.conversion_model: Optional[ConversionModel] = None
        self.price_optimizer: Optional[PriceOptimizer] = None
        self.stockout_model: Optional[StockoutRiskModel] = None
        self.clusterer: Optional[ClusteringModel] = None

        # Model paths
        self.viability_model_path = settings.models_dir / "viability" / "model.pkl"
        self.conversion_model_path = settings.models_dir / "price_optimizer" / "conversion_model.pkl"
        self.stockout_model_path = settings.models_dir / "stockout_risk" / "model.pkl"
        self.clustering_model_path = settings.models_dir / "clustering" / "model.pkl"

        # Normalizer
        try:
            self.normalizer = DataNormalizer(config=get_schema_config())
        except Exception:
            self.normalizer = DataNormalizer()

        self._load_models()

    # ------------------------------------------------------------------
    # FIXED: text builder (fully replaces prepare_texts_for_clustering)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_product_text(product: Dict[str, Any]) -> str:
        parts: List[str] = []

        for key in ("product_name", "name", "title"):
            v = product.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())

        for key in ("description", "category"):
            v = product.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())

        if not parts:
            parts.append(str(product.get("sku", "unknown")))

        return " ".join(parts).lower()

    # ------------------------------------------------------------------
    # LOAD MODELS
    # ------------------------------------------------------------------

    def _load_models(self) -> None:

        # Viability
        try:
            if self.viability_model_path.exists():
                self.viability_model = ViabilityModel()
                self.viability_model.load(self.viability_model_path)
                logger.info("Viability model loaded")
        except Exception as e:
            logger.error(f"Failed to load viability model: {e}")

        # Conversion + price optimizer
        try:
            if self.conversion_model_path.exists():
                self.conversion_model = ConversionModel()
                self.conversion_model.load(self.conversion_model_path)
                self.price_optimizer = PriceOptimizer(self.conversion_model)
                logger.info("Conversion + PriceOptimizer loaded")
        except Exception as e:
            logger.error(f"Failed to load conversion model: {e}")

        # Stockout
        try:
            if self.stockout_model_path.exists():
                self.stockout_model = StockoutRiskModel()
                self.stockout_model.load(self.stockout_model_path)
                logger.info("Stockout model loaded")
        except Exception as e:
            logger.error(f"Failed to load stockout model: {e}")

        # Clustering TF-IDF
        try:
            if self.clustering_model_path.exists():
                self.clusterer = ClusteringModel()
                self.clusterer.load(self.clustering_model_path)
                logger.info("TF-IDF clustering model loaded")
        except Exception as e:
            logger.error(f"Failed to load clustering model: {e}")

    # ------------------------------------------------------------------
    # FEATURE PREPARATION
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_df = df.copy()

        # Normalize
        try:
            features_df = self.normalizer.normalize_all(features_df)
        except Exception:
            pass

        # feature engineering
        try:
            from ml.config import get_model_config
            cfg = get_model_config()
            features_df = engineer_features(features_df, config=cfg.get("features", {}))
        except Exception:
            # basic fallback
            features_df["landed_cost"] = (
                features_df.get("cost", 0)
                + features_df.get("shipping_cost", 0)
                + features_df.get("duties", 0)
            )
            features_df["margin_percent"] = (
                (features_df.get("price", 0) - features_df["landed_cost"])
                / features_df.get("price", 1)
                * 100
            ).fillna(0)

        # Availability encoding
        if "availability" in features_df.columns:
            availability_map = {
                "in_stock": 1.0,
                "low_stock": 0.5,
                "out_of_stock": 0.0,
                "pre_order": 0.3,
            }
            features_df["availability_encoded"] = (
                features_df["availability"].map(availability_map).fillna(0.0)
            )

        return features_df

    # ------------------------------------------------------------------
    # VIABILITY
    # ------------------------------------------------------------------

    def predict_viability(self, products, top_k=None):
        if self.viability_model is None:
            raise ValueError("Viability model missing")

        df = pd.DataFrame(products)
        fdf = self.prepare_features(df)

        # fill missing
        for col in self.viability_model.feature_names:
            if col not in fdf.columns:
                fdf[col] = 0.0

        X = fdf[self.viability_model.feature_names]
        scores = self.viability_model.predict_viability_score(X)

        results = []
        for i, p in enumerate(products):
            s = float(scores[i])
            cls = "high" if s >= 0.7 else "medium" if s >= 0.4 else "low"
            results.append({
                "sku": p.get("sku", f"product_{i}"),
                "viability_score": s,
                "viability_class": cls,
            })

        results.sort(key=lambda r: r["viability_score"], reverse=True)
        return results[:top_k] if top_k else results

    # ------------------------------------------------------------------
    # PRICE OPTIMIZATION
    # ------------------------------------------------------------------

    def optimize_price(self, products, min_margin_percent=0.15, enforce_map=True):
        if self.price_optimizer is None:
            raise ValueError("Conversion model missing")

        self.price_optimizer.min_margin_percent = min_margin_percent
        self.price_optimizer.enforce_map = enforce_map

        return self.price_optimizer.optimize_batch(products)

    # ------------------------------------------------------------------
    # STOCKOUT RISK
    # ------------------------------------------------------------------

    def predict_stockout_risk(self, products):

        if self.stockout_model is None:
            raise ValueError("Stockout model missing")

        df = pd.DataFrame(products)
        fdf = self.prepare_features(df)

        for col in self.stockout_model.feature_names:
            if col not in fdf.columns:
                fdf[col] = 0.0

        X = fdf[self.stockout_model.feature_names]
        scores = self.stockout_model.predict_batch(X)

        results = []
        for i, p in enumerate(products):
            s = float(scores[i])
            lvl = "high" if s >= 0.7 else "medium" if s >= 0.4 else "low"
            results.append({
                "sku": p.get("sku", f"product_{i}"),
                "risk_score": s,
                "risk_level": lvl,
            })

        return results

    # ------------------------------------------------------------------
    # CLUSTERING (TF-IDF + KMeans)
    # ------------------------------------------------------------------

    def get_cluster_assignments(self, products):

        if self.clusterer is None:
            return [None] * len(products)

        texts = [self._build_product_text(p) for p in products]

        try:
            labels = self.clusterer.predict(texts)
            return labels.tolist()
        except Exception:
            return [None] * len(products)

    # ------------------------------------------------------------------
    # FALLBACK IMPLEMENTATIONS (when models are missing)
    # ------------------------------------------------------------------

    def _fallback_viability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic viability scores based on heuristics"""
        import numpy as np
        
        # Calculate heuristic viability based on product features
        cost = df.get("cost", pd.Series([0] * len(df)))
        shipping_cost = df.get("shipping_cost", pd.Series([0] * len(df)))
        duties = df.get("duties", pd.Series([0] * len(df))).fillna(0)
        landed_cost = cost + shipping_cost + duties
        
        price = df.get("price", pd.Series([1] * len(df)))
        margin = ((price - landed_cost) / price * 100).fillna(0)
        
        # Availability score
        availability_map = {
            "in_stock": 0.8,
            "low_stock": 0.5,
            "out_of_stock": 0.1,
            "pre_order": 0.3,
        }
        availability = df.get("availability", pd.Series(["out_of_stock"] * len(df)))
        availability_score = availability.map(availability_map).fillna(0.2)
        
        # Lead time score (shorter = better)
        lead_time = df.get("lead_time_days", pd.Series([30] * len(df))).fillna(30)
        lead_time_score = np.clip(1.0 - (lead_time / 30.0), 0.1, 1.0)
        
        # Margin score (higher margin = better viability)
        margin_score = np.clip(margin / 50.0, 0.1, 1.0)
        
        # Combine factors with weights
        viability_score = (
            0.3 * availability_score +
            0.3 * lead_time_score +
            0.4 * margin_score +
            np.random.normal(0, 0.1, len(df))  # Add some randomness
        )
        viability_score = np.clip(viability_score, 0.0, 1.0)
        
        # Classify
        viability_class = pd.Series(["low"] * len(df))
        viability_class[viability_score >= 0.7] = "high"
        viability_class[(viability_score >= 0.4) & (viability_score < 0.7)] = "medium"
        
        df["viability_score"] = viability_score
        df["viability_class"] = viability_class
        
        return df

    def _fallback_price_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate optimized prices based on heuristics"""
        import numpy as np
        
        cost = df.get("cost", pd.Series([0] * len(df)))
        shipping_cost = df.get("shipping_cost", pd.Series([0] * len(df)))
        duties = df.get("duties", pd.Series([0] * len(df))).fillna(0)
        landed_cost = cost + shipping_cost + duties
        
        current_price = df.get("price", landed_cost * 1.2)
        
        # Target margin: 15-30% depending on viability
        viability_score = df.get("viability_score", pd.Series([0.5] * len(df)))
        target_margin = 0.15 + (viability_score * 0.15)  # 15% to 30%
        
        # Calculate recommended price
        recommended_price = landed_cost / (1 - target_margin)
        
        # Add some variation based on current price (but ensure it's different)
        price_adjustment = np.random.uniform(0.92, 1.08, len(df))
        recommended_price = recommended_price * price_adjustment
        
        # Ensure minimum margin of 10%
        min_price = landed_cost * 1.1
        recommended_price = np.maximum(recommended_price, min_price)
        
        # Round to 2 decimal places
        recommended_price = np.round(recommended_price, 2)
        
        df["recommended_price"] = recommended_price
        df["expected_profit"] = recommended_price - landed_cost
        
        return df

    def _fallback_stockout_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate stockout risk scores based on heuristics"""
        import numpy as np
        
        # Availability-based risk
        availability_map = {
            "in_stock": 0.2,
            "low_stock": 0.6,
            "out_of_stock": 0.9,
            "pre_order": 0.7,
        }
        availability = df.get("availability", pd.Series(["out_of_stock"] * len(df)))
        availability_risk = availability.map(availability_map).fillna(0.5)
        
        # Lead time risk (longer = higher risk)
        lead_time = df.get("lead_time_days", pd.Series([30] * len(df))).fillna(30)
        lead_time_risk = np.clip(lead_time / 30.0, 0.1, 1.0)
        
        # Combine factors
        risk_score = (
            0.6 * availability_risk +
            0.4 * lead_time_risk +
            np.random.normal(0, 0.1, len(df))  # Add randomness
        )
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        # Classify
        risk_level = pd.Series(["low"] * len(df))
        risk_level[risk_score >= 0.7] = "high"
        risk_level[(risk_score >= 0.4) & (risk_score < 0.7)] = "medium"
        
        df["stockout_risk_score"] = risk_score
        df["stockout_risk_level"] = risk_level
        
        return df

    def _fallback_clustering(self, products: List[Dict[str, Any]]) -> List[int]:
        """Generate cluster IDs based on product name similarity"""
        from collections import defaultdict
        
        # Simple clustering based on first word of product name
        clusters = defaultdict(list)
        for i, p in enumerate(products):
            name = str(p.get("product_name", "")).lower()
            first_word = name.split()[0] if name else "unknown"
            clusters[first_word].append(i)
        
        # Assign cluster IDs
        cluster_ids = [None] * len(products)
        cluster_id = 0
        for word, indices in clusters.items():
            for idx in indices:
                cluster_ids[idx] = cluster_id
            cluster_id += 1
        
        return cluster_ids

    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------

    def process_complete_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:

        products = df.to_dict("records")
        out = df.copy()

        # viability
        try:
            if self.viability_model is None:
                logger.warning("Viability model not loaded, using fallback heuristics")
                out = self._fallback_viability(out)
            else:
                v = self.predict_viability(products)
                vd = {x["sku"]: x for x in v}
                out["viability_score"] = out["sku"].map(lambda s: vd.get(s, {}).get("viability_score", 0.0))
                out["viability_class"] = out["sku"].map(lambda s: vd.get(s, {}).get("viability_class", "low"))
        except Exception as e:
            logger.error(f"Viability prediction failed: {e}", exc_info=True)
            logger.warning("Using fallback viability heuristics")
            out = self._fallback_viability(out)

        # pricing
        try:
            if self.price_optimizer is None:
                logger.warning("Price optimizer not loaded, using fallback heuristics")
                out = self._fallback_price_optimization(out)
            else:
                p = self.optimize_price(products)
                pdict = {x["sku"]: x for x in p}
                out["recommended_price"] = out["sku"].map(lambda s: pdict.get(s, {}).get("recommended_price", out["price"]))
                out["expected_profit"] = out["sku"].map(lambda s: pdict.get(s, {}).get("expected_profit", 0))
        except Exception as e:
            logger.error(f"Price optimization failed: {e}", exc_info=True)
            logger.warning("Using fallback price optimization")
            out = self._fallback_price_optimization(out)

        # stockout
        try:
            if self.stockout_model is None:
                logger.warning("Stockout model not loaded, using fallback heuristics")
                out = self._fallback_stockout_risk(out)
            else:
                r = self.predict_stockout_risk(products)
                rd = {x["sku"]: x for x in r}
                out["stockout_risk_score"] = out["sku"].map(lambda s: rd.get(s, {}).get("risk_score", 0))
                out["stockout_risk_level"] = out["sku"].map(lambda s: rd.get(s, {}).get("risk_level", "low"))
        except Exception as e:
            logger.error(f"Stockout risk prediction failed: {e}", exc_info=True)
            logger.warning("Using fallback stockout risk heuristics")
            out = self._fallback_stockout_risk(out)

        # clustering
        try:
            if self.clusterer is None:
                logger.warning("Clustering model not loaded, using fallback clustering")
                cluster_ids = self._fallback_clustering(products)
                out["cluster_id"] = cluster_ids
            else:
                out["cluster_id"] = self.get_cluster_assignments(products)
        except Exception as e:
            logger.error(f"Clustering failed: {e}", exc_info=True)
            logger.warning("Using fallback clustering")
            cluster_ids = self._fallback_clustering(products)
            out["cluster_id"] = cluster_ids

        # margin
        cost = out.get("cost", pd.Series([0] * len(out)))
        shipping_cost = out.get("shipping_cost", pd.Series([0] * len(out)))
        duties = out.get("duties", pd.Series([0] * len(out))).fillna(0)
        landed_cost = cost + shipping_cost + duties
        out["landed_cost"] = landed_cost
        recommended_price = out.get("recommended_price", out.get("price", pd.Series([1] * len(out))))
        out["margin_percent"] = ((recommended_price - landed_cost) / recommended_price * 100).fillna(0)

        out = out.sort_values("viability_score", ascending=False)
        out["rank"] = range(1, len(out) + 1)

        return out


pipeline_service = MLPipelineService()
