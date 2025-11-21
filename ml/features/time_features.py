import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


def create_lead_time_buckets(
    lead_time_days: float,
    buckets: Optional[List[int]] = None
) -> str:
   
    if buckets is None:
        buckets = [0, 7, 14, 21, 30, 60, 90]
    
    if pd.isna(lead_time_days) or lead_time_days < 0:
        return "unknown"
    
    # Find appropriate bucket
    for i in range(len(buckets) - 1):
        if buckets[i] <= lead_time_days < buckets[i + 1]:
            return f"{buckets[i]}-{buckets[i + 1]}"
    
    # If exceeds last bucket
    return f">{buckets[-1]}"


def create_lead_time_category(lead_time_days: float) -> str:
    
    if pd.isna(lead_time_days) or lead_time_days < 0:
        return "unknown"
    
    if lead_time_days <= 7:
        return "very_fast"
    elif lead_time_days <= 14:
        return "fast"
    elif lead_time_days <= 30:
        return "moderate"
    elif lead_time_days <= 60:
        return "slow"
    else:
        return "very_slow"


def get_seasonality_indicator(
    current_date: Optional[date] = None,
    category: Optional[str] = None
) -> Dict[str, Any]:
    
    if current_date is None:
        current_date = date.today()
    
    month = current_date.month
    quarter = (month - 1) // 3 + 1
    
   
    is_holiday_season = month in [11, 12]
    
   
    is_summer = month in [6, 7, 8]
    
   
    is_winter = month in [12, 1, 2]
    
   
    category_seasonality = "neutral"
    if category:
        category_lower = str(category).lower()
        if any(term in category_lower for term in ["summer", "swim", "beach", "outdoor"]):
            category_seasonality = "summer" if is_summer else "off_season"
        elif any(term in category_lower for term in ["winter", "snow", "cold", "heater"]):
            category_seasonality = "winter" if is_winter or month == 12 else "off_season"
        elif any(term in category_lower for term in ["holiday", "christmas", "gift", "decor"]):
            category_seasonality = "holiday" if is_holiday_season else "off_season"
    
    return {
        "month": month,
        "quarter": quarter,
        "is_holiday_season": is_holiday_season,
        "is_summer": is_summer,
        "is_winter": is_winter,
        "category_seasonality": category_seasonality,
    }


def add_time_features(
    df: pd.DataFrame,
    lead_time_col: str = "lead_time_days",
    category_col: Optional[str] = "category",
    current_date: Optional[date] = None,
    lead_time_buckets: Optional[List[int]] = None
) -> pd.DataFrame:
    
    df_features = df.copy()
    
  
    if lead_time_col in df.columns:
      
        df_features["lead_time_bucket"] = df[lead_time_col].apply(
            lambda x: create_lead_time_buckets(x, buckets=lead_time_buckets)
        )
        
       
        df_features["lead_time_category"] = df[lead_time_col].apply(
            create_lead_time_category
        )
        
      
        df_features["is_long_lead_time"] = df[lead_time_col] > 30
        df_features["is_very_long_lead_time"] = df[lead_time_col] > 60
    else:
        logger.warning(f"Lead time column '{lead_time_col}' not found")
        df_features["lead_time_bucket"] = "unknown"
        df_features["lead_time_category"] = "unknown"
        df_features["is_long_lead_time"] = False
        df_features["is_very_long_lead_time"] = False
    
   
    seasonality = get_seasonality_indicator(current_date=current_date)
    
   
    df_features["month"] = seasonality["month"]
    df_features["quarter"] = seasonality["quarter"]
    df_features["is_holiday_season"] = seasonality["is_holiday_season"]
    df_features["is_summer"] = seasonality["is_summer"]
    df_features["is_winter"] = seasonality["is_winter"]
    
        
    if category_col and category_col in df.columns:
        def get_category_seasonality(row):
            category = row.get(category_col)
            return get_seasonality_indicator(current_date=current_date, category=category)["category_seasonality"]
        
        df_features["category_seasonality"] = df.apply(get_category_seasonality, axis=1)
    else:
        df_features["category_seasonality"] = "neutral"
    
    logger.info(f"Added time features: lead_time_bucket, lead_time_category, seasonality indicators")
    
    return df_features

