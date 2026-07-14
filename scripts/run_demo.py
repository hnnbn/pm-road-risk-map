"""Run a lightweight PM risk-scoring demo with bundled sample data.

This demo does not train YOLO, U-Net, or XGBoost. Instead, it uses small
CSV files that mimic model outputs, then runs the same rule-based risk scoring
and Folium map generation used by the full project pipeline.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from road_risk.risk_scoring import add_rule_based_labels  # noqa: E402
from road_risk.visualization import create_risk_map  # noqa: E402


def load_sample_data(sample_dir: Path) -> pd.DataFrame:
    """Load and merge sample damage, crack, and slope feature tables."""
    damage = pd.read_csv(sample_dir / "road_damage_summary.csv")
    crack = pd.read_csv(sample_dir / "crack_metrics.csv")
    slope = pd.read_csv(sample_dir / "slope_features.csv")

    merged = damage.merge(crack, on="image", how="inner")
    merged = merged.merge(slope, on="image", how="inner")
    return merged


def main() -> None:
    sample_dir = PROJECT_ROOT / "data" / "sample"
    output_dir = PROJECT_ROOT / "outputs" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = load_sample_data(sample_dir)
    labeled = add_rule_based_labels(merged)

    csv_path = output_dir / "sample_risk_predictions.csv"
    map_path = output_dir / "sample_risk_map.html"

    labeled.to_csv(csv_path, index=False, encoding="utf-8-sig")
    create_risk_map(labeled, map_path)

    print("Demo complete")
    print(f"Rows: {len(labeled)}")
    print("Risk grade distribution:")
    print(labeled["final_risk_grade"].value_counts().sort_index().to_string())
    print(f"Saved CSV: {csv_path}")
    print(f"Saved map: {map_path}")


if __name__ == "__main__":
    main()
