# Data

This repository does not include raw AI-Hub imagery or DEM files because of size and license constraints.

Expected working layout for the notebooks:

```text
data/
  raw/
    road_damage_annotations/
    road_damage_images/
    seoul_road_images/
    dem/
  processed/
    yolo/
    crack_masks/
    risk_table.csv
```

In the original Colab workflow, the same folders were mounted from Google Drive under `/content/drive/MyDrive/...`.
