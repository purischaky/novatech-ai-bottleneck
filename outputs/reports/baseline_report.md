# 📊 Baseline Model Report

**Model** RandomForest (n_estimators=200)
**Date:** 2026-09-08 12:41

## 🎯 Clasification: SLA Breach

| Metric | Value |
|---------|-------|
| accuracy | 0.9045 |
| precision | 0.9052 |
| recall | 0.9139 |
| f1 | 0.9095 |
| roc_auc | 0.9712 |

## 📈 Regression: Resolution Hours

| Metric | Value |
|---------|-------|
| mae | 20.2464 |
| rmse | 32.0142 |
| r2 | 0.6416 |

## 🏆 Top 10 Features (Clasification)

| Feature | Importance |
|---------|-------------|
| num__waiting_hours | 0.2314 |
| num__queue_age_hours | 0.2201 |
| num__sla_hours | 0.1097 |
| num__first_response_hours | 0.1064 |
| cat__assigned_team_Tier 2 | 0.0670 |
| cat__assigned_team_Tier 3 | 0.0452 |
| cat__priority_Low | 0.0406 |
| cat__priority_High | 0.0309 |
| cat__category_Technical Issue | 0.0304 |
| cat__category_Product Question | 0.0189 |
