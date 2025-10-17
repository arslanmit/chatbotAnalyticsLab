from src.models.core import DatasetType
from src.repositories.persistence import (
    ConversationRepository,
    DatasetRepository,
    ExperimentRepository,
)
from tests.factories import sample_dataset


def test_conversation_repository_roundtrip():
    repo = ConversationRepository()
    dataset = sample_dataset()
    repo.save_dataset_conversations(dataset)

    loaded = repo.load_conversations(dataset.name, dataset.dataset_type)
    assert len(loaded) == 1
    conv = loaded[0]
    assert conv.id == "conv-1"
    assert conv.turns[0].text == "Hello"
    assert conv.turns[0].intent == "greet"


def test_experiment_repository_save_and_list():
    experiment_repo = ExperimentRepository()
    summary = {
        "run_id": "run-1",
        "pipeline_config": {"model_id": "model-1"},
        "dataset": {"type": DatasetType.BANKING77.value, "name": "TestDataset", "path": "/tmp/test"},
        "training_metrics": {"accuracy": 0.9},
        "validation_metrics": {"accuracy": 0.85},
        "test_metrics": {"accuracy": 0.8},
        "model_artifact_path": "/tmp/model",
        "tags": ["test"],
    }
    experiment_repo.save_from_summary(summary)
    runs = experiment_repo.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == "run-1"


def test_dataset_repository_upsert():
    dataset_repo = DatasetRepository()
    dataset_repo.save(
        name="Data1",
        dataset_type=DatasetType.BANKING77.value,
        path="/tmp/dataset",
        metadata={"meta": True},
    )
    record = dataset_repo.get("Data1", DatasetType.BANKING77.value)
    assert record is not None
    assert record.path == "/tmp/dataset"
    assert record.metadata_json["meta"] is True
