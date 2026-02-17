# Contract Digitization AI

End-to-end AI system for:
1. Extracting full contract content from selectable and non-selectable PDFs in page order.
2. Classifying product images by appearance (fridge, HVAC, etc.).
3. Feedback-driven retraining with model promotion gates.

## Core Features
- Hybrid PDF extraction for selectable + scanned documents.
- Layout-aware reconstruction report with side-by-side original vs extracted structure.
- Product image classifier (CLIP zero-shot baseline + trainable head).
- Automatic low-confidence feedback queue.
- Retraining pipeline with validation metrics and promotion policy.

## Project Layout
- `/Users/apple/ml-setup/ContractDigitization/src/contract_ai/contracts`: contract extraction pipeline
- `/Users/apple/ml-setup/ContractDigitization/src/contract_ai/classification`: image classification
- `/Users/apple/ml-setup/ContractDigitization/src/contract_ai/retraining`: feedback and retraining
- `/Users/apple/ml-setup/ContractDigitization/src/contract_ai/mlops`: MLOps utilities (gates, tracking, flows, monitoring)
- `/Users/apple/ml-setup/ContractDigitization/src/contract_ai/api`: FastAPI service
- `/Users/apple/ml-setup/ContractDigitization/scripts/mlops`: MLOps scripts
- `/Users/apple/ml-setup/ContractDigitization/data`: runtime outputs, models, logs

## Requirements
- Python 3.10+
- Tesseract OCR installed locally
- Optional MLOps extras: DVC, MLflow UI, Prefect

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[dev,mlops]
```

## CLI Usage
### 1) Extract contract PDF
```bash
contract-ai extract-contract --input data/incoming/sample_contract.pdf --output-dir data/exports/sample_contract
```

### 2) Classify product image
```bash
contract-ai classify-image --input data/incoming/product.jpg --labels fridge,hvac,washing_machine,microwave
```

### 3) Add feedback
```bash
contract-ai add-feedback --input data/incoming/product.jpg --true-label hvac
```

### 4) Retrain with gates
```bash
contract-ai retrain
```

### 5) Run Prefect retrain flow
```bash
contract-ai mlops-run-flow --min-samples 20 --device cpu
```

## API Usage
```bash
uvicorn contract_ai.api.main:app --reload
```
Endpoints:
- `POST /contracts/extract`
- `POST /classification/predict`
- `POST /feedback/classification`
- `POST /retrain/run`

## MLOps Components Added
### 1) Data/model versioning (DVC)
Files:
- `/Users/apple/ml-setup/ContractDigitization/dvc.yaml`
- `/Users/apple/ml-setup/ContractDigitization/params.yaml`

Initialize once:
```bash
dvc init
```

(Optional) connect S3 remote:
```bash
dvc remote add -d storage s3://<your-bucket>/contract-ai-dvc
dvc remote modify storage region <aws-region>
```

Run pipeline:
```bash
dvc repro
```

### 2) Experiment tracking (MLflow)
Retraining logs metrics and artifacts to MLflow (`CONTRACT_AI_MLFLOW_TRACKING_URI`).

Start local UI:
```bash
mlflow ui --backend-store-uri ./data/mlruns --port 5000
```

### 3) Orchestration (Prefect)
Flow file:
- `/Users/apple/ml-setup/ContractDigitization/src/contract_ai/mlops/flows.py`

Run:
```bash
python scripts/mlops/run_prefect_flow.py --min-samples 20 --device cpu
```

### 4) CI/CD (GitHub Actions)
- CI: `/Users/apple/ml-setup/ContractDigitization/.github/workflows/ci.yml`
- AWS ECS deploy: `/Users/apple/ml-setup/ContractDigitization/.github/workflows/deploy-ecs.yml`

### 5) Promotion policy (quality gates)
Implemented in:
- `/Users/apple/ml-setup/ContractDigitization/src/contract_ai/mlops/gates.py`

Model is promoted only if weighted metrics pass thresholds:
- `promote_min_f1`
- `promote_min_precision`
- `promote_min_recall`

### 6) Monitoring logs
Prediction logs:
- `/Users/apple/ml-setup/ContractDigitization/data/monitoring/predictions.jsonl`

Summary job:
```bash
python scripts/mlops/summarize_monitoring.py
```

### 7) Environment-specific config
- `/Users/apple/ml-setup/ContractDigitization/configs/dev.yaml`
- `/Users/apple/ml-setup/ContractDigitization/configs/staging.yaml`
- `/Users/apple/ml-setup/ContractDigitization/configs/prod.yaml`

## What to Verify for MLOps
1. Reproducibility:
- `dvc repro` completes and regenerates expected outputs.

2. Tracking:
- MLflow run created for retrain, with params/metrics/artifacts.

3. Promotion gates:
- Low-quality run is blocked (`promoted=false`).
- Good run updates `/data/models/classifier/current/classifier.joblib`.

4. Data lineage:
- Feedback data and model version metadata are traceable.

5. Monitoring:
- Predictions are appended to `predictions.jsonl`.
- Monitoring summary reflects real distribution and average confidence.

6. CI/CD:
- CI workflow passes lint/tests/docker build.
- Deploy workflow pushes image and updates ECS service.

7. Runtime health:
- `/health` endpoint is stable after deploy.
