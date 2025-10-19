# Chatbot Analytics Lab

End-to-end experimentation environment for exploring chatbot conversation datasets, training intent classifiers, and visualizing performance. The project bundles a FastAPI service, Streamlit dashboard, data processing utilities, and monitoring hooks so teams can ingest conversation logs, iterate on models, and track outcomes in one place.

## Features
- **Dataset ingestion** – Load banking-focused datasets (BANKING77, Bitext, Schema-Guided, etc.), validate them, and persist summaries via the API.
- **Training pipeline** – Orchestrated workflow for preprocessing, splitting, training BERT-based intent classifiers, evaluating, and versioning artifacts.
- **Experiment tracking** – Store metrics, confusion matrices, and run metadata for later inspection in the dashboard.
- **Interactive dashboard** – Streamlit app for experiments, intent distributions, conversation flows, and sentiment trends with CSV/PDF exports.
- **Monitoring & alerts** – Request latency metrics, system health probes, and pluggable alert channels (log/webhook) for threshold breaches.
- **Docker-native workflow** – Dockerfiles and compose stack to run the API and dashboard together with minimal setup.

## Architecture
- `src/api`: FastAPI application exposing dataset, conversation, intent, and training endpoints with middleware for rate limiting and metrics.
- `src/services`: Core business logic (data preprocessing, validation, model training, performance analysis, backups, hyperparameter search).
- `src/models`: Shared data models, dataset abstractions, and the BERT-powered `IntentClassifier`.
- `dashboard/`: Streamlit front-end that consumes persisted analytics and experiment results.
- `src/repositories`: Persistence layer for datasets, conversations, experiments, and trained model artifacts.
- `src/monitoring`: System and request monitoring utilities plus alert channel integrations.

SQLite (default) stores metadata, while model artifacts and processed data land under `./models` and `./data`.

## Repository Layout
```
chatbotAnalyticsLab/
├── src/                  # Application source (API, services, utilities)
├── dashboard/            # Streamlit entry point
├── Dataset/              # Place raw datasets here (BANKING77, Bitext, etc.)
├── data/                 # Generated processed data, caches, and SQLite DB volume
├── models/               # Saved model checkpoints and tokenizer assets
├── tests/                # Pytest suite (API, monitoring, persistence, backups)
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # API + dashboard stack definition
└── Makefile              # Helper commands for Docker workflows
```

## Getting Started
1. **Prerequisites**
   - Python 3.10+
   - Optional: Docker 24+ if you plan to run the stack in containers

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure settings (optional)**
   - Default values live in `src/config/settings.py` and `config.json`.
   - Override by exporting environment variables (e.g. `DATABASE_URL`, `DEFAULT_MODEL`, `DATASET_DIR`) before running services.

## Running the Services
### Initialize the workspace
Populate required directories and the SQLite schema, then verify configuration:
```bash
python -m src.main
```

### FastAPI application
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```
- Interactive docs: `http://localhost:8000/docs`
- Key routes: `/datasets/upload`, `/conversations/{id}`, `/intents/predict`, `/training/run`, `/training/hyperparameter-search`, `/health/live`

### Streamlit dashboard
Ensure the API and database artifacts are accessible, then launch:
```bash
streamlit run dashboard/app.py
```
- Provides overviews, experiment history, intent distributions, conversation flow analytics, and export buttons (CSV/PDF).

### Docker Compose (API + dashboard)
Build and run both services:
```bash
docker compose up --build
```
Volumes map `./data`, `./logs`, and `./models` into the containers for persistence. Adjust ports or resource limits in `docker-compose.yml`.

## Working With Datasets & Models
- Place raw datasets under `Dataset/` (e.g. `Dataset/BANKING77/`).
- Upload via API:
  ```bash
  curl -X POST http://localhost:8000/datasets/upload \
    -H "Content-Type: application/json" \
    -d '{"dataset_type": "banking77", "file_path": "Dataset/BANKING77", "preprocess": true}'
  ```
- Trigger training (runs synchronously by default):
  ```bash
  curl -X POST http://localhost:8000/training/run \
    -H "Content-Type: application/json" \
    -d '{"dataset_type": "banking77", "training_overrides": {"num_epochs": 3}}'
  ```
- Trained artifacts, evaluation reports, and run logs are stored under `./pipeline_runs/<model_id_timestamp>/` and registered through the experiment tracker.

## Monitoring & Alerts
- Request metrics and latencies are collected via middleware and stored under `app.state.metrics_collector`.
- `src/monitoring/system.py` samples CPU/memory; thresholds come from `settings.alerts`.
- Configure alert channels by setting `ALERT_CHANNELS` (comma-separated, e.g. `log,webhook:https://hooks.example.com`).

## Testing
```bash
pytest
```
The suite covers API integration, persistence, backup/restore logic, alerting, and performance helpers. Add new tests under `tests/` mirroring the module namespace.

## Helpful Commands
- `make build-api` / `make build-dashboard` – build respective Docker images.
- `make compose-up` / `make compose-down` – manage the docker-compose stack.
- `make clean-images` / `make clean-volumes` – tidy up local Docker resources.

## Troubleshooting
- Missing datasets: check that the folder structure matches your `DatasetType` value (`Dataset/banking77`, `Dataset/bitext`, etc.).
- Model downloads: set `MODEL_CACHE_DIR` to a writable location if running inside restricted environments.
- Long training runs: switch to async execution by posting `{"async_run": true}` to `/training/run` and monitor progress via experiment logs.
