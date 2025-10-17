# Chatbot Analytics System

## Development Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m pytest
```

## Running with Docker Compose

```bash
docker compose build
docker compose up
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

Persisted data/logs/models are mounted from the host `data/`, `logs/`, and `models/` directories.
