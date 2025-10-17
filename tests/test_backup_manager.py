from src.models.core import DatasetType
from src.repositories.persistence import ConversationRepository
from src.services.backup_manager import BackupManager
from tests.factories import sample_dataset


def test_backup_and_restore(tmp_path):
    repo = ConversationRepository()
    dataset = sample_dataset("BackupDataset")
    repo.save_dataset_conversations(dataset)

    manager = BackupManager(backup_dir=str(tmp_path / "backups"))

    backup_path = manager.backup_dataset(dataset.name, dataset.dataset_type)
    assert backup_path.exists()

    manager.restore_dataset("RestoredDataset", dataset.dataset_type, backup_path)
    restored = repo.load_conversations("RestoredDataset", dataset.dataset_type)
    assert len(restored) == 1
    assert restored[0].turns[0].text == "Hello"
