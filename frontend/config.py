import os
from pathlib import Path
from typing import Optional


class FrontendConfig:
   
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_TIMEOUT: int = 300  # seconds
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    
 
    PAGE_TITLE: str = "DropSmart"
    PAGE_ICON: str = "📦"
    
   
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list[str] = [".xlsx", ".xls"]
    
  
    MAX_ROWS_DISPLAY: int = 1000
    DEFAULT_SORT_COLUMN: str = "viability_score"


# Global config instance
config = FrontendConfig()

