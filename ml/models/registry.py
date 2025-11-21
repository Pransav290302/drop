from pathlib import Path

from ml.models.viability_model import ViabilityModel
from ml.models.price_model import ConversionModel       # FIXED (correct file)
from ml.models.stockout_model import StockoutRiskModel  # FIXED (correct file)
from ml.models.clustering_model import ClusteringModel


class ModelRegistry:
   
    def __init__(self, root: str = "data/models"):
        self.root = Path(root)

        # instantiate empty objects
        self.viability = ViabilityModel()
        self.conversion = ConversionModel()
        self.stockout = StockoutRiskModel()
        self.clustering = ClusteringModel()

    def load_all(self):

        viability_path = self.root / "viability" / "model.pkl"
        self.viability.load(viability_path)


        conv_path = self.root / "price_optimizer" / "conversion_model.pkl"
        self.conversion.load(conv_path)


        stockout_path = self.root / "stockout_risk" / "model.pkl"
        self.stockout.load(stockout_path)

 
        cluster_path = self.root / "clustering" / "kmeans.pkl"
        self.clustering.load(cluster_path)

        print("🔥 All ML models loaded successfully!")

    def get_all(self):
       
        return {
            "viability": self.viability,
            "conversion": self.conversion,
            "stockout": self.stockout,
            "clustering": self.clustering,
        }
