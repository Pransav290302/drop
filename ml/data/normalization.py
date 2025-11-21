import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from ml.config import get_schema_config

logger = logging.getLogger(__name__)


class DataNormalizer:
  
    def __init__(self, config: Optional[Dict[str, Any]] = None):
       
        if config is None:
            schema_config = get_schema_config()
            config = schema_config
        
        self.config = config
        
        # Currency conversion rates
        currency_config = config.get("currency", {})
        self.default_currency = currency_config.get("default", "USD")
        self.conversion_rates = currency_config.get("conversion_rates", {"USD": 1.0})
        
        # Unit configurations
        units_config = config.get("units", {})
        self.default_weight_unit = units_config.get("weight", {}).get("default", "kg")
        self.default_dimension_unit = units_config.get("dimension", {}).get("default", "cm")
        
        logger.info(f"Initialized DataNormalizer with default currency: {self.default_currency}")
    
    def normalize_currency(
        self,
        df: pd.DataFrame,
        amount_columns: List[str],
        currency_column: Optional[str] = None,
        target_currency: str = "USD"
    ) -> pd.DataFrame:
       
        df_normalized = df.copy()
        
      
        if currency_column is None or currency_column not in df.columns:
            
            logger.info(f"Assuming all amounts are in {self.default_currency}")
            source_currency = self.default_currency
        else:
       
            unique_currencies = df[currency_column].unique()
            source_currency = unique_currencies[0] if len(unique_currencies) == 1 else None
        
       
        for col in amount_columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping currency normalization")
                continue
            
          
            if currency_column and currency_column in df.columns:
                
                def convert_row(row):
                    source_curr = str(row.get(currency_column, self.default_currency)).upper()
                    amount = row.get(col, 0.0)
                    
                    if pd.isna(amount) or amount == 0:
                        return amount
                    
                    
                    source_rate = self.conversion_rates.get(source_curr, 1.0)
                    target_rate = self.conversion_rates.get(target_currency.upper(), 1.0)
                    
                    
                    if source_rate > 0:
                        converted = amount * (target_rate / source_rate)
                        return converted
                    return amount
                
                df_normalized[col] = df.apply(convert_row, axis=1)
            else:
                
                source_rate = self.conversion_rates.get(source_currency.upper(), 1.0)
                target_rate = self.conversion_rates.get(target_currency.upper(), 1.0)
                
                if source_rate > 0 and source_rate != target_rate:
                    conversion_factor = target_rate / source_rate
                    df_normalized[col] = df[col] * conversion_factor
                    logger.info(f"Converted {col} from {source_currency} to {target_currency} (factor: {conversion_factor:.4f})")
        
        return df_normalized
    
    def normalize_weight(
        self,
        df: pd.DataFrame,
        weight_column: str,
        weight_unit_column: Optional[str] = None,
        target_unit: str = "kg"
    ) -> pd.DataFrame:
        """
        Normalize weight values to target unit.
        
        Supported units: kg, g, lb, oz
        
        Args:
            df: DataFrame with weight values
            weight_column: Column name containing weight
            weight_unit_column: Column name indicating unit (if None, assumes default)
            target_unit: Target unit for conversion (default: kg)
            
        Returns:
            DataFrame with normalized weight values
        """
        df_normalized = df.copy()
        
        if weight_column not in df.columns:
            logger.warning(f"Weight column '{weight_column}' not found")
            return df_normalized
        
       
        to_kg = {
            "kg": 1.0,
            "g": 0.001,
            "lb": 0.453592,
            "oz": 0.0283495,
        }
        
        
        from_kg = {
            "kg": 1.0,
            "g": 1000.0,
            "lb": 2.20462,
            "oz": 35.274,
        }
        
        if weight_unit_column and weight_unit_column in df.columns:
            
            def convert_weight(row):
                source_unit = str(row.get(weight_unit_column, self.default_weight_unit)).lower()
                weight = row.get(weight_column, 0.0)
                
                if pd.isna(weight) or weight == 0:
                    return weight
                
                
                source_to_kg = to_kg.get(source_unit, 1.0)
                kg_to_target = from_kg.get(target_unit.lower(), 1.0)
                
                weight_kg = weight * source_to_kg
                weight_target = weight_kg * kg_to_target
                
                return weight_target
            
            df_normalized[weight_column] = df.apply(convert_weight, axis=1)
        else:
           
            source_unit = self.default_weight_unit.lower()
            source_to_kg = to_kg.get(source_unit, 1.0)
            kg_to_target = from_kg.get(target_unit.lower(), 1.0)
            
            if source_to_kg != 1.0 or kg_to_target != 1.0:
                conversion_factor = source_to_kg * kg_to_target
                df_normalized[weight_column] = df[weight_column] * conversion_factor
                logger.info(f"Converted {weight_column} from {source_unit} to {target_unit}")
        
        return df_normalized
    
    def normalize_dimension(
        self,
        df: pd.DataFrame,
        dimension_columns: List[str],
        dimension_unit_column: Optional[str] = None,
        target_unit: str = "cm"
    ) -> pd.DataFrame:
        """
        Normalize dimension values to target unit.
        
        Supported units: cm, m, in, ft
        
        Args:
            df: DataFrame with dimension values
            dimension_columns: List of column names containing dimensions
            dimension_unit_column: Column name indicating unit (if None, assumes default)
            target_unit: Target unit for conversion (default: cm)
            
        Returns:
            DataFrame with normalized dimension values
        """
        df_normalized = df.copy()
        
       
        to_cm = {
            "cm": 1.0,
            "m": 100.0,
            "in": 2.54,
            "ft": 30.48,
        }
        

        from_cm = {
            "cm": 1.0,
            "m": 0.01,
            "in": 0.393701,
            "ft": 0.0328084,
        }
        
        for col in dimension_columns:
            if col not in df.columns:
                logger.warning(f"Dimension column '{col}' not found, skipping")
                continue
            
            if dimension_unit_column and dimension_unit_column in df.columns:
               
                def convert_dimension(row):
                    source_unit = str(row.get(dimension_unit_column, self.default_dimension_unit)).lower()
                    dimension = row.get(col, 0.0)
                    
                    if pd.isna(dimension) or dimension == 0:
                        return dimension
                    
                   
                    source_to_cm = to_cm.get(source_unit, 1.0)
                    cm_to_target = from_cm.get(target_unit.lower(), 1.0)
                    
                    dimension_cm = dimension * source_to_cm
                    dimension_target = dimension_cm * cm_to_target
                    
                    return dimension_target
                
                df_normalized[col] = df.apply(convert_dimension, axis=1)
            else:
               
                source_unit = self.default_dimension_unit.lower()
                source_to_cm = to_cm.get(source_unit, 1.0)
                cm_to_target = from_cm.get(target_unit.lower(), 1.0)
                
                if source_to_cm != 1.0 or cm_to_target != 1.0:
                    conversion_factor = source_to_cm * cm_to_target
                    df_normalized[col] = df[col] * conversion_factor
                    logger.info(f"Converted {col} from {source_unit} to {target_unit}")
        
        return df_normalized
    
    def normalize_all(
        self,
        df: pd.DataFrame,
        currency_columns: Optional[List[str]] = None,
        weight_column: Optional[str] = None,
        dimension_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        
        df_normalized = df.copy()
        
       
        if currency_columns is None:
            currency_columns = ["cost", "price", "shipping_cost", "duties", "map_price"]
            currency_columns = [col for col in currency_columns if col in df.columns]
        
       
        if currency_columns:
            df_normalized = self.normalize_currency(
                df_normalized,
                currency_columns,
                target_currency=self.default_currency
            )
        
       
        if weight_column:
            if weight_column in df.columns:
                df_normalized = self.normalize_weight(
                    df_normalized,
                    weight_column,
                    target_unit=self.default_weight_unit
                )
        

        if dimension_columns is None:
            dimension_columns = ["length_cm", "width_cm", "height_cm"]
            dimension_columns = [col for col in dimension_columns if col in df.columns]
        
       
        if dimension_columns:
            df_normalized = self.normalize_dimension(
                df_normalized,
                dimension_columns,
                target_unit=self.default_dimension_unit
            )
        
        logger.info("Data normalization complete")
        
        return df_normalized


def normalize_dataframe(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
   
    normalizer = DataNormalizer(config)
    return normalizer.normalize_all(df)

