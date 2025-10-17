# Deployment Guide

## Requirements

- Docker 20+
- docker compose v2
- Python 3.12 (for local testing)

## Build & Run

```bash
docker compose build
docker compose up -d
```

Services:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

## Environment Variables

Key variables (set in `docker-compose.yml` or environment):
- `DATABASE_URL`: Database connection (default sqlite:///data/chatbot_analytics.db)
- `DATASET_DIR`: Dataset location inside container
- `MODEL_CACHE_DIR`: Model cache path
- `ALERT_CHANNELS`: Comma-separated alert channels (e.g., `log`, `webhook:https://...`)
- `STREAMLIT_SERVER_*`: Streamlit configuration

## Testing Before Deploy

```bash
python3 -m pytest
```

## Backup & Restore

Use `BackupManager` (Python API) to export or restore datasets; ensure `/data` volume persists.
