"""Backup manager for datasets and conversations."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.settings import settings
from src.repositories.persistence import ConversationRepository
from src.models.core import DatasetType
from src.utils.files import ensure_dir, rotate_files
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BackupManager:
    """Handle dataset backups and restoration."""

    def __init__(self, backup_dir: Optional[str] = None):
        self.backup_dir = ensure_dir(backup_dir or settings.data.backup_dir)
        self.format = settings.data.backup_format.lower()
        self.retention = settings.data.backup_retention
        self.conversation_repo = ConversationRepository()

    def backup_dataset(self, dataset_name: str, dataset_type: DatasetType) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dataset_dir = self.backup_dir / dataset_type.value / dataset_name
        ensure_dir(dataset_dir)

        conversations = self.conversation_repo.load_conversations(dataset_name, dataset_type)
        if not conversations:
            raise ValueError(f"No conversations found for dataset {dataset_name}")

        backup_path = dataset_dir / f"{timestamp}.{self.format}"
        if self.format == "json":
            self._backup_json(conversations, backup_path)
        elif self.format == "parquet":
            self._backup_parquet(conversations, backup_path)
        else:
            raise ValueError(f"Unsupported backup format: {self.format}")

        rotate_files(dataset_dir, self.retention)
        logger.info("Backed up dataset %s (%s) to %s", dataset_name, dataset_type.value, backup_path)
        return backup_path

    def restore_dataset(self, dataset_name: str, dataset_type: DatasetType, backup_path: Path) -> None:
        if self.format == "json" or backup_path.suffix.lower() == ".json":
            with open(backup_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.conversation_repo.save_dataset_conversations(_conversations_to_dataset(dataset_name, dataset_type, data))
        elif self.format == "parquet" or backup_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(backup_path)
            self.conversation_repo.save_dataset_conversations(_parquet_to_dataset(dataset_name, dataset_type, df))
        else:
            raise ValueError(f"Unsupported backup format: {backup_path.suffix}")
        logger.info("Restored dataset %s (%s) from %s", dataset_name, dataset_type.value, backup_path)

    def list_backups(self, dataset_name: str, dataset_type: DatasetType) -> list[Path]:
        dataset_dir = self.backup_dir / dataset_type.value / dataset_name
        if not dataset_dir.exists():
            return []
        return sorted(dataset_dir.glob(f"*.{self.format}"), key=lambda p: p.stat().st_mtime, reverse=True)

    def cleanup(self) -> None:
        for dataset_dir in self.backup_dir.glob("*/**"):
            if dataset_dir.is_dir():
                rotate_files(dataset_dir, self.retention)


def _conversations_to_dataset(dataset_name: str, dataset_type: DatasetType, data: list[dict]) -> "Dataset":
    from src.models.core import Conversation, ConversationTurn, Dataset, Speaker

    conversations = []
    for conversation_data in data:
        turns = [
            ConversationTurn(
                speaker=Speaker(turn["speaker"]),
                text=turn["text"],
                timestamp=datetime.fromisoformat(turn["timestamp"]) if turn.get("timestamp") else None,
                intent=turn.get("intent"),
                confidence=turn.get("confidence"),
            )
            for turn in conversation_data["turns"]
        ]
        conversations.append(
            Conversation(
                id=conversation_data["id"],
                turns=turns,
                source_dataset=dataset_type,
                metadata=conversation_data.get("metadata", {}),
                success=conversation_data.get("success"),
            )
        )

    return Dataset(
        name=dataset_name,
        dataset_type=dataset_type,
        conversations=conversations,
        metadata={"restored_from": dataset_name},
    )


def _parquet_to_dataset(dataset_name: str, dataset_type: DatasetType, df: pd.DataFrame) -> "Dataset":
    from src.models.core import Conversation, ConversationTurn, Dataset, Speaker

    conversations = []
    for conversation_id, group in df.groupby("conversation_id"):
        turns = []
        for _, row in group.sort_values("order_index").iterrows():
            timestamp = row["timestamp"]
            turns.append(
                ConversationTurn(
                    speaker=Speaker(row["speaker"]),
                    text=row["text"],
                    timestamp=datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else None,
                    intent=row.get("intent"),
                    confidence=row.get("confidence"),
                )
            )
        conversations.append(
            Conversation(
                id=conversation_id,
                turns=turns,
                source_dataset=dataset_type,
                metadata={},
                success=group["success"].iloc[0] if "success" in group else None,
            )
        )

    return Dataset(
        name=dataset_name,
        dataset_type=dataset_type,
        conversations=conversations,
        metadata={"restored_from": dataset_name},
    )


def _backup_json(conversations, path: Path) -> None:
    serializable = [
        {
            "id": conv.id,
            "success": conv.success,
            "metadata": conv.metadata,
            "turns": [
                {
                    "speaker": turn.speaker.value if hasattr(turn.speaker, "value") else turn.speaker,
                    "text": turn.text,
                    "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
                    "intent": turn.intent,
                    "confidence": turn.confidence,
                }
                for turn in conv.turns
            ],
        }
        for conv in conversations
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def _backup_parquet(conversations, path: Path) -> None:
    rows = []
    for conv in conversations:
        for index, turn in enumerate(conv.turns):
            rows.append(
                {
                    "conversation_id": conv.id,
                    "order_index": index,
                    "success": conv.success,
                    "speaker": turn.speaker.value if hasattr(turn.speaker, "value") else turn.speaker,
                    "text": turn.text,
                    "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
                    "intent": turn.intent,
                    "confidence": turn.confidence,
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
