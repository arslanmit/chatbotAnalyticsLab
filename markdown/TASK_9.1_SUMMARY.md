# Task 9.1 – Performance Monitoring

## Highlights

- Added psutil-backed system monitoring utilities with execution timing decorators (`src/monitoring/system.py`, `src/monitoring/__init__.py`).
- Extended health metrics endpoint to surface live CPU, memory, and disk usage alongside request stats (`src/api/routes/health.py`, `src/api/schemas/common.py`).
- Instrumented dataset ingestion and training pipeline loaders with timing hooks to capture processing durations (`src/api/routes/datasets.py`, `src/services/training_pipeline.py`).

## Key Outputs

- Centralized `collect_system_metrics()` enabling API/CLI access to process + system resource data.
- `/health/metrics` now returns both request metrics and system telemetry for external monitoring.
- Requirements updated (`psutil`) to support ongoing performance diagnostics.
