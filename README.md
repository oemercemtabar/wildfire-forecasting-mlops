# Wildfire Risk Forecasting System (MLOps)
## Predicting environmental crises through reproducible, automated machine learning.

## Project Overview
Wildfires represent one of the most significant threats to global ecosystems and human safety. This project implements an end-to-end MLOps Pipeline to forecast wildfire risks based on real-time meteorological data (temperature, humidity, wind speed) and historical satellite imagery.

Unlike a static notebook, this repository functions as a production-ready "Early Warning Factory." It automates the entire lifecycle—from data ingestion and validation to model deployment and "Climate Drift" monitoring.

---

## The 4-Phase MLOps Architecture

### 1.Data Engineering & Lineage (Phase 1)
- Data Sources: Integration with NASA FIRMS (satellite fire data) and OpenWeather API.
- Versioning: Managed via DVC (Data Version Control) to ensure that every model experiment is tied to the exact dataset version used, preventing "data mystery."
- Validation: Automated checks for "sensor failure" values (e.g., impossible temperatures) before training begins.
### 2.Automated Experimentation (Phase 2)
- Tracking: Every training run is logged using MLflow, capturing hyperparameters, loss curves, and feature importance (e.g., how much "Fuel Moisture" impacted the risk score).
- Registry: High-performing models are versioned in a central Model Registry, allowing for seamless rollbacks if a new model underperforms.
### 3.CI/CD & Orchestration (Phase 3)
- Automation: Using GitHub Actions to run unit tests and model signature validations on every push.
- Orchestration: The pipeline ensures that data cleaning, feature engineering, and training happen in a strict, reproducible sequence defined in dvc.yaml.
### 4.Serving & Reliability (Phase 4)
- Deployment: A high-performance FastAPI wrapper containerized with Docker for cloud-agnostic deployment.
- Monitoring: Integrated with Evidently AI to detect "Climate Drift." If the current environment deviates significantly from the training distribution (e.g., unprecedented heatwaves), the system triggers an alert for manual review or retraining.

--- 
## Tech Stack
- Language: Python 3.9+
- ML Frameworks: Scikit-Learn / XGBoost (Risk Classification)
- Ops Tools: DVC, MLflow, Docker
- API: FastAPI, Uvicorn
- CI/CD: GitHub Actions
- Monitoring: Evidently AI

---

## Impact Goal
To provide emergency responders and environmental agencies with a reliable, verifiable, and transparent tool for resource allocation, moving away from "black-box" models toward auditable ML systems.