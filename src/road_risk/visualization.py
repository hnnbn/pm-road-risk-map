"""Map visualization helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import folium
except ImportError:  # pragma: no cover - fallback for lightweight demos
    folium = None


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
    if folium is None:
        return create_static_risk_report(df, output_html, lat_col, lon_col, grade_col)

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


def create_static_risk_report(
    df: pd.DataFrame,
    output_html: str | Path,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    grade_col: str = "final_risk_grade",
):
    """Create a dependency-light HTML report when Folium is unavailable."""
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in df.dropna(subset=[lat_col, lon_col]).iterrows():
        grade = int(row[grade_col])
        color = RISK_COLORS.get(grade, "#636363")
        label = RISK_LABELS.get(grade, str(grade))
        rows.append(
            "<tr>"
            f"<td>{row.get('image', '')}</td>"
            f"<td><span class='dot' style='background:{color}'></span>{label}</td>"
            f"<td>{row.get('final_risk_score', ''):.3f}</td>"
            f"<td>{row[lat_col]:.5f}</td>"
            f"<td>{row[lon_col]:.5f}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PM Risk Demo Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #555; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 24px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .dot {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
  </style>
</head>
<body>
  <h1>PM Risk Demo Report</h1>
  <p>Folium is not installed, so this static table was generated instead of an interactive map.</p>
  <table>
    <thead>
      <tr><th>Image</th><th>Risk Grade</th><th>Score</th><th>Latitude</th><th>Longitude</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    return None
