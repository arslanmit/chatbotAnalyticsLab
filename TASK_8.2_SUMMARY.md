# Task 8.2 – Data Persistence Layer

## Highlights

- Expanded the SQLAlchemy ORM to include conversation and turn tables with dataset relationships and indexes, enabling granular analytics storage (`src/repositories/orm.py`).
- Added repository helpers to persist and retrieve conversations, experiments, and model artifacts, with JSON-friendly serialization and speaker/timestamp handling (`src/repositories/persistence.py`, `src/services/experiment_tracker.py`, `src/repositories/model_repository.py`).
- Hooked persistence into dataset ingestion so uploaded datasets populate metadata and conversation records automatically, while exports now ignore non-Latin characters in reports (`src/api/routes/datasets.py`, `src/dashboard/exporter.py`).

## Key Outputs

- Database now mirrors experiment runs, model versions, dataset metadata, and conversation transcripts for efficient querying.
- Train and dataset flows write to SQLite on initialization; `init_db()` ensures schema creation at startup.
- Repository exports updated for reuse across services and future API endpoints.
