import os
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import json
import logging

logger = logging.getLogger(__name__)

class DatasetEDA:
    """Automated Exploratory Data Analysis (EDA) engine for deepfake datasets."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def analyze(self) -> Dict[str, Any]:
        return self.run_analysis()

    def run_analysis(self) -> Dict[str, Any]:
        """Scans the dataset directory and computes EDA statistics."""
        stats = {
            "dataset_path": str(self.dataset_path),
            "exists": self.dataset_path.exists(),
            "total_images": 0,
            "classes": {
                "real": 0,
                "fake": 0
            },
            "formats": {},
            "dimensions": [],
            "corrupt_files": [],
            "class_balance": {}
        }

        if not self.dataset_path.exists():
            logger.warning(f"Dataset path '{self.dataset_path}' does not exist.")
            return stats

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

        for split in ["train", "validation", "test", "."]:
            for cls in ["real", "fake", "deepfake"]:
                target_dir = self.dataset_path / split / cls if split != "." else self.dataset_path / cls
                if target_dir.exists():
                    class_key = "fake" if cls in ["fake", "deepfake"] else "real"
                    for file_path in target_dir.rglob("*"):
                        if file_path.suffix.lower() in valid_exts:
                            stats["total_images"] += 1
                            stats["classes"][class_key] += 1
                            
                            # Inspect image
                            try:
                                with Image.open(file_path) as img:
                                    fmt = (img.format or "UNKNOWN").upper()
                                    stats["formats"][fmt] = stats["formats"].get(fmt, 0) + 1
                                    w, h = img.size
                                    stats["dimensions"].append((w, h))
                            except Exception as e:
                                stats["corrupt_files"].append(str(file_path))

        # Calculate class balance
        total = stats["total_images"]
        if total > 0:
            stats["class_balance"] = {
                "real_percent": round((stats["classes"]["real"] / total) * 100, 2),
                "fake_percent": round((stats["classes"]["fake"] / total) * 100, 2)
            }

        return stats

    def generate_report(self, output_file: str = "eda_report.json") -> str:
        """Generates JSON report of the EDA analysis."""
        stats = self.run_analysis()
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"EDA report generated and saved to '{out_path}'.")
        return str(out_path)

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset"
    eda = DatasetEDA(path)
    report_file = eda.generate_report()
    print(f"EDA report successfully saved to {report_file}")
