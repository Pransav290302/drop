import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_volumetric_weight(
    length_cm: float,
    width_cm: float,
    height_cm: float,
    divisor: float = 5000.0
) -> float:
   
    if pd.isna(length_cm) or pd.isna(width_cm) or pd.isna(height_cm):
        return 0.0
    
    if length_cm <= 0 or width_cm <= 0 or height_cm <= 0:
        return 0.0
    
    volumetric_weight = (length_cm * width_cm * height_cm) / divisor
    return volumetric_weight


def classify_size_tier(
    length_cm: float,
    width_cm: float,
    height_cm: float,
    weight_kg: float,
    tiers: Optional[Dict[str, Dict[str, float]]] = None
) -> str:
    
    if tiers is None:
        # Default tier definitions
        tiers = {
            "small": {
                "max_length": 30,
                "max_width": 30,
                "max_height": 30,
                "max_weight": 1.0,
            },
            "medium": {
                "max_length": 60,
                "max_width": 60,
                "max_height": 60,
                "max_weight": 5.0,
            },
            "large": {
                "max_length": 120,
                "max_width": 120,
                "max_height": 120,
                "max_weight": 20.0,
            },
        }
    
   
    if pd.isna(length_cm) or pd.isna(width_cm) or pd.isna(height_cm):
        # Use weight only if dimensions missing
        if not pd.isna(weight_kg):
            if weight_kg <= tiers["small"]["max_weight"]:
                return "small"
            elif weight_kg <= tiers["medium"]["max_weight"]:
                return "medium"
            elif weight_kg <= tiers["large"]["max_weight"]:
                return "large"
            else:
                return "oversized"
        return "unknown"
    

    max_dimension = max(length_cm, width_cm, height_cm)
    
   
    if (max_dimension <= tiers["small"]["max_length"] and 
        (pd.isna(weight_kg) or weight_kg <= tiers["small"]["max_weight"])):
        return "small"
    elif (max_dimension <= tiers["medium"]["max_length"] and 
          (pd.isna(weight_kg) or weight_kg <= tiers["medium"]["max_weight"])):
        return "medium"
    elif (max_dimension <= tiers["large"]["max_length"] and 
          (pd.isna(weight_kg) or weight_kg <= tiers["large"]["max_weight"])):
        return "large"
    else:
        return "oversized"


def add_weight_features(
    df: pd.DataFrame,
    length_col: str = "length_cm",
    width_col: str = "width_cm",
    height_col: str = "height_cm",
    weight_col: str = "weight_kg",
    volumetric_divisor: float = 5000.0
) -> pd.DataFrame:
   
    df_features = df.copy()
    
   
    has_dimensions = (
        length_col in df.columns and
        width_col in df.columns and
        height_col in df.columns
    )
    
    has_weight = weight_col in df.columns
    
    if not has_dimensions and not has_weight:
        logger.warning("No dimension or weight columns found, skipping weight features")
        return df_features
    
   
    if has_dimensions:
        df_features["volumetric_weight"] = df.apply(
            lambda row: calculate_volumetric_weight(
                row.get(length_col, 0),
                row.get(width_col, 0),
                row.get(height_col, 0),
                divisor=volumetric_divisor
            ),
            axis=1
        )
        
       
        df_features["volume_cm3"] = (
            df_features[length_col] *
            df_features[width_col] *
            df_features[height_col]
        ).fillna(0)
        
       
        df_features["max_dimension"] = df_features[[length_col, width_col, height_col]].max(axis=1)
    else:
        df_features["volumetric_weight"] = 0.0
        df_features["volume_cm3"] = 0.0
        df_features["max_dimension"] = 0.0
    
   
    if has_dimensions or has_weight:
        df_features["size_tier"] = df.apply(
            lambda row: classify_size_tier(
                row.get(length_col, 0) if has_dimensions else 0,
                row.get(width_col, 0) if has_dimensions else 0,
                row.get(height_col, 0) if has_dimensions else 0,
                row.get(weight_col, 0) if has_weight else 0,
            ),
            axis=1
        )
    else:
        df_features["size_tier"] = "unknown"
    
        
    if has_weight and has_dimensions:
        df_features["billable_weight"] = df_features[[weight_col, "volumetric_weight"]].max(axis=1)
    elif has_weight:
        df_features["billable_weight"] = df_features[weight_col]
    elif has_dimensions:
        df_features["billable_weight"] = df_features["volumetric_weight"]
    else:
        df_features["billable_weight"] = 0.0
    
    logger.info(f"Added weight features: volumetric_weight, size_tier, volume_cm3, max_dimension, billable_weight")
    
    return df_features

