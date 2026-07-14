# GitHub Upload Guide

현재 상위 작업 폴더에는 다른 미추적 파일도 있으므로, 이 프로젝트만 올리려면 `pm-road-risk-map` 폴더만 선택해서 커밋하세요.

## Option A. Add this folder to the current repository

```bash
git add pm-road-risk-map
git commit -m "Add PM road risk map portfolio project"
git push
```

## Option B. Create a separate GitHub repository

```bash
cd pm-road-risk-map
git init
git add .
git commit -m "Initial portfolio project"
git branch -M main
git remote add origin https://github.com/<your-id>/pm-road-risk-map.git
git push -u origin main
```

The raw dataset, trained weights, and generated outputs are intentionally excluded. Put large artifacts in Google Drive, Hugging Face, or GitHub Releases if you need to share them.
