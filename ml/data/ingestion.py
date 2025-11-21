import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_excel_file(file_path: Path) -> pd.DataFrame:
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Try reading Excel file
        df = pd.read_excel(file_path, engine='openpyxl')
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise ValueError(f"Failed to load Excel file: {e}")


def validate_dataframe(df: pd.DataFrame) -> bool:
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if len(df) == 0:
        raise ValueError("DataFrame has no rows")
    
    return True

