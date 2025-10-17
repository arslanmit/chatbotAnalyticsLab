# API Documentation (Quick Reference)

## Health
- `GET /health`
- `GET /health/metrics`
- `GET /health/alerts?trigger=false`

## Datasets
- `POST /datasets/upload` (JSON body with dataset path/type; supports preprocessing)

## Intents
- `POST /intents/predict`
- `POST /intents/predict/batch`

## Conversations
- `POST /conversations/analyze`
- `POST /conversations/trends`

## Training
- `POST /training/run`
- `POST /training/hyperparameter-search`

Refer to FastAPI docs at `/docs` for detailed schemas.
