# Task 11.1 – Docker Containerization

## Highlights

- Added production-ready Dockerfiles for the FastAPI API (`Dockerfile.api`) and Streamlit dashboard (`Dockerfile.dashboard`) with slim Python base images.
- Created a `docker-compose.yml` to run both services with shared volumes for data, logs, and models, plus environment overrides for database and alerts.
- Introduced a `.dockerignore` and updated README with build/run instructions for Docker workflows.

## Key Outputs

- `docker compose build && docker compose up` launches API on port 8000 and dashboard on port 8501.
- Configuration supports persistent storage (SQLite DB + datasets) via mounted host directories.
- Ready foundation for deployment documentation and environment-specific configuration (Task 11.2).
