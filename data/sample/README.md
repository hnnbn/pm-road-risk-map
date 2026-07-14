# Sample Data

This folder contains a tiny synthetic sample dataset that mimics the output schema of the full project pipeline.

The sample is intentionally small so reviewers can run the risk-scoring and map-generation code without downloading the original AI-Hub imagery, DEM files, or trained model weights.

## Files

| File | Description |
|---|---|
| `road_damage_summary.csv` | Image-level pothole/manhole features from YOLO predictions |
| `crack_metrics.csv` | Image-level crack features from U-Net mask predictions |
| `slope_features.csv` | GPS coordinates and DEM-derived slope values |

## Join Key

All files use the `image` column as the join key.

## Note

The values are demo-friendly examples, not original project data.
