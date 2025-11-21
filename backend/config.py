import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    
    
  
    app_name: str = "DropSmart API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
   
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    

    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    models_dir: Path = data_dir / "models"
    outputs_dir: Path = data_dir / "outputs"
    

    max_file_size_mb: int = 50
    max_file_size_bytes: int = 50 * 1024 * 1024
    allowed_extensions: list[str] = [".xlsx", ".xls"]

    max_skus: int = 10000
    processing_timeout_seconds: int = 300
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False



settings = Settings()


settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.outputs_dir.mkdir(parents=True, exist_ok=True)

