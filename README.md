# PM Road Risk Map

멀티모달 융합 기반 PM(Personal Mobility) 주행 위험도 예측 및 안전 지도 시각화 프로젝트입니다. 도로 노면 이미지에서 포트홀·맨홀·크랙을 감지하고, GPS 기반 DEM 경사도 정보를 결합해 서울 도로 이미지별 위험 등급을 산출했습니다.

## Project Summary

개인형 이동장치(PM)는 작은 바퀴와 낮은 안정성 때문에 포트홀, 균열, 급경사에 민감합니다. 이 프로젝트는 이미지 기반 노면 손상과 지형 기반 경사도를 함께 사용해 PM 주행 위험도를 예측하고 지도에 시각화하는 파이프라인을 구성합니다.

전체 흐름은 다음과 같습니다.

1. AI-Hub 도로 이미지 annotation을 YOLO 형식으로 변환
2. YOLOv8로 포트홀·맨홀 객체 탐지
3. U-Net으로 크랙 영역 segmentation
4. 서울 도로 이미지의 GPS와 DEM을 결합해 경사도 추출
5. 손상 등급, 크랙 등급, 경사 등급으로 rule-based risk label 생성
6. XGBoost로 최종 위험 등급 예측
7. Folium 기반 PM 위험 지도 생성

## Repository Structure

```text
pm-road-risk-map/
  configs/
    yolo_pothole_manhole.yaml
  data/
    README.md
  docs/
    final_presentation.pdf
  notebooks/
    01_pothole_manhole_yolov8.ipynb
    02_crack_unet_segmentation.ipynb
    03_risk_score_and_map.ipynb
  outputs/
    README.md
  src/
    road_risk/
      pothole_manhole.py
      crack_unet.py
      risk_scoring.py
      visualization.py
  requirements.txt
```

## Methods

### Pothole and Manhole Detection

- Model: YOLOv8n / YOLOv8s detection
- Labels: `category_id=8` pothole, `category_id=10` manhole
- Improvement: 수도권 데이터에서 포트홀 label mismatch를 확인한 뒤 수도권 외 포트홀 데이터를 재수집·검수했습니다.
- Result from presentation: pothole mAP50 improved from `0.0939` to `0.476`, and overall mAP50 improved from `0.484` to `0.703`.

### Crack Segmentation

- Initial attempt: YOLOv8 segmentation
- Final model: U-Net with ResNet encoder
- Loss: Dice Loss + Focal Loss
- Reason for U-Net: 크랙은 얇고 긴 polyline 구조라 객체 polygon 방식보다 pixel-level binary mask 학습이 적합했습니다.

### Rule-Based Risk Label

The final risk score follows the presentation logic:

```python
final_risk_score = (
    0.45 * damage_grade
    + 0.35 * crack_grade
    + 0.20 * slope_grade
    + 0.4 * I(damage_grade >= 2 and slope_grade >= 2)
    + 0.3 * I(crack_grade >= 2 and slope_grade >= 2)
    + 0.3 * I(damage_grade >= 2 and crack_grade >= 2)
)
```

Risk grade:

| Grade | Score Range | Meaning |
|---:|---|---|
| 0 | <= 0.75 | Safe |
| 1 | 0.75 - 1.5 | Caution |
| 2 | 1.5 - 2.25 | Danger |
| 3 | > 2.25 | Very dangerous |

## Quick Start

```bash
pip install -r requirements.txt
```

In Colab, open the notebooks in order:

1. `notebooks/01_pothole_manhole_yolov8.ipynb`
2. `notebooks/02_crack_unet_segmentation.ipynb`
3. `notebooks/03_risk_score_and_map.ipynb`

Raw AI-Hub images, labels, DEM files, trained weights, and generated outputs are intentionally excluded from Git. Put local files under `data/` or mount Google Drive paths as shown in the notebooks.

## Portfolio Notes

This repository is organized for portfolio review rather than direct redistribution of the original dataset. The notebooks preserve the Colab workflow, while reusable logic is collected under `src/road_risk` so reviewers can understand the modeling and scoring pipeline without reading one long notebook.
