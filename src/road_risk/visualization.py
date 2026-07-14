"""Map visualization helpers."""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd


RISK_COLORS = {
    0: "#2ca25f",
    1: "#fdae61",
    2: "#f46d43",
    3: "#a50026",
}

RISK_LABELS = {
    0: "safe",
    1: "caution",
    2: "danger",
    3: "very dangerous",
}


def create_risk_map(
    df: pd.DataFrame,
    output_html: str | Path,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    grade_col: str = "final_risk_grade",
) -> folium.Map:
    """Create and save a Folium map from image-level risk predictions."""
    center = [df[lat_col].mean(), df[lon_col].mean()]
    fmap = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    for _, row in df.dropna(subset=[lat_col, lon_col]).iterrows():
        grade = int(row[grade_col])
        popup = folium.Popup(
            html=(
                f"<b>{row.get('image', '')}</b><br>"
                f"risk: {RISK_LABELS.get(grade, grade)}<br>"
                f"score: {row.get('final_risk_score', '')}"
            ),
            max_width=260,
        )
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=5 + grade,
            color=RISK_COLORS.get(grade, "#636363"),
            fill=True,
            fill_opacity=0.75,
            popup=popup,
        ).add_to(fmap)

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_html))
    return fmap
