"""Rule-based scores used to create labels for the PM risk model.

The formulas are organized from the final presentation:

- road damage: pothole/manhole object count and area ratio
- crack: predicted crack mask area, skeleton length, and connected components
- slope: DEM-derived slope angle
- final risk: weighted multimodal grade with interaction bonuses
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def grade_damage(score: float) -> int:
    """Map pothole/manhole damage score to 0-3 grade."""
    if score == 0:
        return 0
    if score < 5:
        return 1
    if score < 15:
        return 2
    return 3


def calculate_damage_score(
    pothole_count: float = 0,
    manhole_count: float = 0,
    pothole_area_ratio: float = 0,
    max_pothole_area_ratio: float = 0,
) -> float:
    """Calculate a compact road-surface damage score.

    Area ratio is weighted highest because road maintenance standards commonly
    treat damaged area as a stronger indicator than raw object count.
    """
    return (
        1.0 * pothole_count
        + 0.3 * manhole_count
        + 1000.0 * pothole_area_ratio
        + 500.0 * max_pothole_area_ratio
    )


def grade_crack(score: float) -> int:
    """Map crack score to 0-3 grade."""
    if score < 5:
        return 0
    if score < 15:
        return 1
    if score < 30:
        return 2
    return 3


def calculate_crack_score(
    crack_area_ratio: float,
    crack_length_ratio: float,
    component_count: float,
) -> float:
    """Calculate crack severity from predicted U-Net masks."""
    return (
        1000.0 * crack_area_ratio
        + 10.0 * crack_length_ratio
        + 0.2 * min(component_count, 50)
    )


def grade_slope(slope_degree: float) -> int:
    """Map slope angle in degrees to 0-3 grade.

    Boundaries follow the presentation's degree equivalents for 3%, 5%, and 7%
    longitudinal grades.
    """
    if slope_degree <= 1.7:
        return 0
    if slope_degree <= 2.9:
        return 1
    if slope_degree <= 4.0:
        return 2
    return 3


def calculate_final_risk_score(
    damage_grade: int,
    crack_grade: int,
    slope_grade: int,
) -> float:
    """Calculate multimodal PM risk score with interaction bonuses."""
    damage_grade = int(damage_grade)
    crack_grade = int(crack_grade)
    slope_grade = int(slope_grade)

    score = 0.45 * damage_grade + 0.35 * crack_grade + 0.20 * slope_grade
    score += 0.4 if damage_grade >= 2 and slope_grade >= 2 else 0
    score += 0.3 if crack_grade >= 2 and slope_grade >= 2 else 0
    score += 0.3 if damage_grade >= 2 and crack_grade >= 2 else 0
    return score


def grade_final_risk(score: float) -> int:
    """Map final risk score to 0-3 grade."""
    if score <= 0.75:
        return 0
    if score <= 1.5:
        return 1
    if score <= 2.25:
        return 2
    return 3


def add_rule_based_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add damage, crack, slope, and final risk labels to a dataframe."""
    out = df.copy()

    out["damage_score"] = [
        calculate_damage_score(
            row.get("pothole_count", 0),
            row.get("manhole_count", 0),
            row.get("pothole_area_ratio", 0),
            row.get("max_pothole_area_ratio", 0),
        )
        for _, row in out.iterrows()
    ]
    out["damage_grade"] = out["damage_score"].map(grade_damage)

    out["crack_score"] = [
        calculate_crack_score(
            row.get("crack_area_ratio", 0),
            row.get("crack_length_ratio", 0),
            row.get("component_count", 0),
        )
        for _, row in out.iterrows()
    ]
    out["crack_grade"] = out["crack_score"].map(grade_crack)

    out["slope_grade"] = out["slope_degree"].map(grade_slope)
    out["final_risk_score"] = [
        calculate_final_risk_score(d, c, s)
        for d, c, s in zip(out["damage_grade"], out["crack_grade"], out["slope_grade"])
    ]
    out["final_risk_grade"] = out["final_risk_score"].map(grade_final_risk)
    return out


def class_weights(y: pd.Series) -> dict[int, float]:
    """Return inverse-frequency class weights for imbalanced risk labels."""
    counts = y.value_counts().sort_index()
    total = counts.sum()
    n_classes = len(counts)
    return {int(cls): float(total / (n_classes * count)) for cls, count in counts.items()}
