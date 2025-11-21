import requests
from typing import Dict, Any, Optional, List
import logging
from frontend.config import config

logger = logging.getLogger(__name__)


class APIClient:
    
    
    def __init__(self, base_url: Optional[str] = None):
        
        self.base_url = base_url or config.API_BASE_URL
        self.timeout = config.API_TIMEOUT
    
    def upload_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        
        url = f"{self.base_url}/api/v1/upload"
        
        # Validate filename
        if not filename:
            raise ValueError("Filename is required")
        
        # Ensure filename has proper extension
        if not filename.lower().endswith(('.xlsx', '.xls')):
            raise ValueError(f"File must have .xlsx or .xls extension. Got: {filename}")
        
        files = {"file": (filename, file_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        
        try:
            response = requests.post(url, files=files, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
           
            error_detail = "Unknown error"
            try:
                if response.text:
                    error_data = response.json()
                    error_detail = error_data.get("detail", str(e))
                else:
                    error_detail = response.text or str(e)
            except:
                error_detail = str(e)
            
            logger.error(f"Failed to upload file: {error_detail}")
            raise Exception(f"Upload failed: {error_detail}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to upload file: {e}")
            raise Exception(f"Upload failed: {str(e)}")
    
    def validate_schema(self, file_id: str) -> Dict[str, Any]:
        
        url = f"{self.base_url}/api/v1/validate"
        
        payload = {"file_id": file_id}
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to validate schema: {e}")
            raise
    
    def predict_viability(
        self,
        products: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        
        url = f"{self.base_url}/api/v1/predict_viability"
        
        params = {}
        if top_k is not None:
            params["top_k"] = top_k
        
        payload = {"products": products}
        
        try:
            response = requests.post(url, json=payload, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to predict viability: {e}")
            raise
    
    def optimize_price(
        self,
        products: List[Dict[str, Any]],
        min_margin_percent: float = 0.15,
        enforce_map: bool = True
    ) -> Dict[str, Any]:
        
        url = f"{self.base_url}/api/v1/optimize_price"
        
        payload = {
            "products": products,
            "min_margin_percent": min_margin_percent,
            "enforce_map": enforce_map,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to optimize price: {e}")
            raise
    
    def predict_stockout_risk(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        url = f"{self.base_url}/api/v1/stockout_risk"
        
        payload = {"products": products}
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to predict stockout risk: {e}")
            raise
    
    def get_results(self, file_id: str) -> Dict[str, Any]:
        
        url = f"{self.base_url}/api/v1/get_results"
        
        params = {"file_id": file_id}
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get results: {e}")
            raise
    
    def export_csv(self, file_id: str) -> bytes:
        
        url = f"{self.base_url}/api/v1/export_csv"
        
        params = {"file_id": file_id}
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to export CSV: {e}")
            raise
    
    def health_check(self) -> bool:
        
        url = f"{self.base_url}/health"
        
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


# Global API client instance
api_client = APIClient()

