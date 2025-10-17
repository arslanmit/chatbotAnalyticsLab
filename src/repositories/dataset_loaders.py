"""
Dataset loaders for various banking conversation datasets.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional

from src.models.core import (
    Dataset,
    Conversation,
    ConversationTurn,
    DatasetType,
    Speaker,
)
from src.interfaces.base import DatasetLoaderInterface
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Banking77Loader(DatasetLoaderInterface):
    """Loader for BANKING77 dataset (JSON format with intent labels)."""

    def __init__(self):
        self.intent_labels: Optional[List[str]] = None

    def validate_format(self, path: Path) -> bool:
        """Validate if the dataset format is supported."""
        if not path.exists():
            return False

        # Check for required files
        data_dir = path / "data" if path.is_dir() else path.parent
        required_files = ["train.json", "test.json", "intent_labels.json"]

        return all((data_dir / f).exists() for f in required_files)

    def _load_intent_labels(self, data_dir: Path) -> List[str]:
        """Load intent label mappings."""
        labels_file = data_dir / "intent_labels.json"

        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                labels_data = json.load(f)
                # Handle both list and dict formats
                if isinstance(labels_data, list):
                    return labels_data
                elif isinstance(labels_data, dict):
                    return list(labels_data.values())
                else:
                    raise ValueError(
                        f"Unexpected format for intent labels: {type(labels_data)}"
                    )
        except Exception as e:
            logger.warning(f"Could not load intent labels: {e}. Using numeric labels.")
            return []

    def load(self, path: Path) -> Dataset:
        """Load BANKING77 dataset from the given path."""
        if not self.validate_format(path):
            raise ValueError(f"Invalid BANKING77 dataset format at {path}")

        data_dir = path / "data" if path.is_dir() else path.parent

        # Load intent labels
        self.intent_labels = self._load_intent_labels(data_dir)

        conversations = []

        # Load train, validation, and test splits
        for split_name in ["train", "val", "test"]:
            split_file = data_dir / f"{split_name}.json"
            if split_file.exists():
                conversations.extend(self._load_split(split_file, split_name))

        logger.info(f"Loaded {len(conversations)} conversations from BANKING77 dataset")

        return Dataset(
            name="BANKING77",
            dataset_type=DatasetType.BANKING77,
            conversations=conversations,
            metadata={
                "num_intents": len(self.intent_labels) if self.intent_labels else 77,
                "intent_labels": self.intent_labels,
                "source": "PolyAI-LDN/task-specific-datasets",
            },
        )

    def _load_split(self, file_path: Path, split_name: str) -> List[Conversation]:
        """Load a single split file."""
        conversations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for idx, item in enumerate(data):
                # Extract text and label
                text = item.get("text", "")
                label_idx = item.get("label", -1)

                # Get intent name from label index
                if self.intent_labels and 0 <= label_idx < len(self.intent_labels):
                    intent = self.intent_labels[label_idx]
                else:
                    intent = f"intent_{label_idx}"

                # Create a single-turn conversation
                turn = ConversationTurn(
                    speaker=Speaker.USER, text=text, intent=intent, confidence=1.0
                )

                conversation = Conversation(
                    id=f"banking77_{split_name}_{idx}",
                    turns=[turn],
                    source_dataset=DatasetType.BANKING77,
                    metadata={"split": split_name, "label_idx": label_idx},
                )

                conversations.append(conversation)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

        return conversations


class BitextLoader(DatasetLoaderInterface):
    """Loader for Bitext Retail Banking dataset (CSV/Parquet format)."""

    def validate_format(self, path: Path) -> bool:
        """Validate if the dataset format is supported."""
        if not path.exists():
            return False

        # Check for CSV or Parquet files
        if path.is_file():
            return path.suffix in [".csv", ".parquet"]

        # Check directory for CSV/Parquet files
        csv_files = list(path.glob("*.csv"))
        parquet_files = list(path.glob("*.parquet"))

        return len(csv_files) > 0 or len(parquet_files) > 0

    def load(self, path: Path) -> Dataset:
        """Load Bitext dataset from the given path."""
        if not self.validate_format(path):
            raise ValueError(f"Invalid Bitext dataset format at {path}")

        # Determine file to load
        if path.is_file():
            file_path = path
        else:
            # Prefer CSV over Parquet
            csv_files = list(path.glob("*.csv"))
            if csv_files:
                file_path = csv_files[0]
            else:
                parquet_files = list(path.glob("*.parquet"))
                file_path = parquet_files[0]

        logger.info(f"Loading Bitext dataset from {file_path}")

        if file_path.suffix == ".csv":
            conversations = self._load_csv(file_path)
        else:
            conversations = self._load_parquet(file_path)

        logger.info(f"Loaded {len(conversations)} conversations from Bitext dataset")

        return Dataset(
            name="BitextRetailBanking",
            dataset_type=DatasetType.BITEXT,
            conversations=conversations,
            metadata={
                "source": "Bitext Retail Banking LLM Chatbot Training Dataset",
                "file": str(file_path),
            },
        )

    def _load_csv(self, file_path: Path) -> List[Conversation]:
        """Load Bitext CSV file."""
        conversations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for idx, row in enumerate(reader):
                    # Extract fields
                    instruction = row.get("instruction", "")
                    response = row.get("response", "")
                    category = row.get("category", "")
                    intent = row.get("intent", "")
                    tags = row.get("tags", "")

                    # Create conversation turns
                    turns = [
                        ConversationTurn(
                            speaker=Speaker.USER, text=instruction, intent=intent
                        ),
                        ConversationTurn(speaker=Speaker.ASSISTANT, text=response),
                    ]

                    conversation = Conversation(
                        id=f"bitext_{idx}",
                        turns=turns,
                        source_dataset=DatasetType.BITEXT,
                        metadata={"category": category, "intent": intent, "tags": tags},
                    )

                    conversations.append(conversation)

        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            raise

        return conversations

    def _load_parquet(self, file_path: Path) -> List[Conversation]:
        """Load Bitext Parquet file."""
        try:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(file_path)
            conversations = []

            for idx, row in df.iterrows():
                instruction = row.get("instruction", "")
                response = row.get("response", "")
                category = row.get("category", "")
                intent = row.get("intent", "")
                tags = row.get("tags", "")

                turns = [
                    ConversationTurn(
                        speaker=Speaker.USER, text=instruction, intent=intent
                    ),
                    ConversationTurn(speaker=Speaker.ASSISTANT, text=response),
                ]

                conversation = Conversation(
                    id=f"bitext_{idx}",
                    turns=turns,
                    source_dataset=DatasetType.BITEXT,
                    metadata={"category": category, "intent": intent, "tags": tags},
                )

                conversations.append(conversation)

            return conversations

        except ImportError:
            logger.error(
                "pandas is required to load Parquet files. Install with: pip install pandas pyarrow"
            )
            raise
        except Exception as e:
            logger.error(f"Error loading Parquet {file_path}: {e}")
            raise


class SchemaGuidedLoader(DatasetLoaderInterface):
    """Loader for Schema-Guided Dialogue dataset (JSON format with multi-turn dialogues)."""

    def validate_format(self, path: Path) -> bool:
        """Validate if the dataset format is supported."""
        if not path.exists():
            return False

        # Check for dialogue JSON files
        if path.is_file() and path.suffix == ".json":
            return True

        # Check directory for dialogue files
        if path.is_dir():
            json_files = list(path.glob("dialogues_*.json"))
            return len(json_files) > 0

        return False

    def load(self, path: Path) -> Dataset:
        """Load Schema-Guided dataset from the given path."""
        if not self.validate_format(path):
            raise ValueError(f"Invalid Schema-Guided dataset format at {path}")

        conversations = []

        if path.is_file():
            conversations.extend(self._load_dialogue_file(path))
        else:
            # Load all dialogue files in directory
            json_files = sorted(path.glob("dialogues_*.json"))
            for json_file in json_files:
                conversations.extend(self._load_dialogue_file(json_file))

        logger.info(
            f"Loaded {len(conversations)} conversations from Schema-Guided dataset"
        )

        return Dataset(
            name="SchemaGuidedDialogue",
            dataset_type=DatasetType.SCHEMA_GUIDED,
            conversations=conversations,
            metadata={
                "source": "Google Schema-Guided Dialogue Dataset",
                "domain": "banking",
            },
        )

    def _load_dialogue_file(self, file_path: Path) -> List[Conversation]:
        """Load a single dialogue JSON file."""
        conversations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                dialogues = json.load(f)

            for dialogue in dialogues:
                dialogue_id = dialogue.get("dialogue_id", "")
                services = dialogue.get("services", [])
                turns_data = dialogue.get("turns", [])

                turns = []
                for turn_data in turns_data:
                    speaker_str = turn_data.get("speaker", "USER")
                    utterance = turn_data.get("utterance", "")
                    frames = turn_data.get("frames", [])

                    # Extract intent from frames
                    intent = None
                    if frames and len(frames) > 0:
                        frame = frames[0]
                        state = frame.get("state", {})
                        intent = state.get("active_intent", None)

                    speaker = (
                        Speaker.USER if speaker_str == "USER" else Speaker.ASSISTANT
                    )

                    turn = ConversationTurn(
                        speaker=speaker,
                        text=utterance,
                        intent=intent if speaker == Speaker.USER else None,
                    )

                    turns.append(turn)

                conversation = Conversation(
                    id=f"schema_guided_{dialogue_id}",
                    turns=turns,
                    source_dataset=DatasetType.SCHEMA_GUIDED,
                    metadata={"dialogue_id": dialogue_id, "services": services},
                )

                conversations.append(conversation)

        except Exception as e:
            logger.error(f"Error loading dialogue file {file_path}: {e}")
            raise

        return conversations


class TwitterSupportLoader(DatasetLoaderInterface):
    """Loader for Customer Support on Twitter dataset (CSV format)."""

    def validate_format(self, path: Path) -> bool:
        """Validate if the dataset format is supported."""
        if not path.exists():
            return False

        if path.is_file() and path.suffix == ".csv":
            return True

        if path.is_dir():
            csv_files = list(path.glob("*.csv"))
            return len(csv_files) > 0

        return False

    def load(self, path: Path) -> Dataset:
        """Load Twitter Support dataset from the given path."""
        if not self.validate_format(path):
            raise ValueError(f"Invalid Twitter Support dataset format at {path}")

        if path.is_file():
            file_path = path
        else:
            csv_files = list(path.glob("*.csv"))
            file_path = csv_files[0]

        logger.info(f"Loading Twitter Support dataset from {file_path}")

        conversations = self._load_csv(file_path)

        logger.info(
            f"Loaded {len(conversations)} conversations from Twitter Support dataset"
        )

        return Dataset(
            name="CustomerSupportTwitter",
            dataset_type=DatasetType.TWITTER_SUPPORT,
            conversations=conversations,
            metadata={"source": "Customer Support on Twitter", "file": str(file_path)},
        )

    def _load_csv(self, file_path: Path) -> List[Conversation]:
        """Load Twitter Support CSV file."""
        conversations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # Group tweets by conversation
                conversation_map: Dict[str, List[Dict]] = {}

                for row in reader:
                    # Extract fields (adjust based on actual CSV structure)
                    tweet_id = row.get("tweet_id", "")
                    in_response_to = row.get("in_response_to_tweet_id", "")
                    text = row.get("text", "")
                    author_id = row.get("author_id", "")

                    # Determine conversation ID
                    conv_id = in_response_to if in_response_to else tweet_id

                    if conv_id not in conversation_map:
                        conversation_map[conv_id] = []

                    conversation_map[conv_id].append(
                        {
                            "tweet_id": tweet_id,
                            "text": text,
                            "author_id": author_id,
                            "in_response_to": in_response_to,
                        }
                    )

                # Convert to Conversation objects
                for idx, (conv_id, tweets) in enumerate(conversation_map.items()):
                    turns = []

                    for tweet in tweets:
                        # Determine speaker (simple heuristic)
                        is_support = (
                            "support" in tweet["author_id"].lower()
                            or not tweet["in_response_to"]
                        )
                        speaker = Speaker.ASSISTANT if is_support else Speaker.USER

                        turn = ConversationTurn(speaker=speaker, text=tweet["text"])

                        turns.append(turn)

                    conversation = Conversation(
                        id=f"twitter_{conv_id}",
                        turns=turns,
                        source_dataset=DatasetType.TWITTER_SUPPORT,
                        metadata={
                            "conversation_id": conv_id,
                            "num_tweets": len(tweets),
                        },
                    )

                    conversations.append(conversation)

        except Exception as e:
            logger.error(f"Error loading Twitter Support CSV {file_path}: {e}")
            raise

        return conversations


class SyntheticSupportLoader(DatasetLoaderInterface):
    """Loader for Synthetic Tech Support dataset (CSV format)."""

    def validate_format(self, path: Path) -> bool:
        """Validate if the dataset format is supported."""
        if not path.exists():
            return False

        if path.is_file() and path.suffix == ".csv":
            return True

        if path.is_dir():
            csv_files = list(path.glob("*.csv"))
            return len(csv_files) > 0

        return False

    def load(self, path: Path) -> Dataset:
        """Load Synthetic Support dataset from the given path."""
        if not self.validate_format(path):
            raise ValueError(f"Invalid Synthetic Support dataset format at {path}")

        if path.is_file():
            file_path = path
        else:
            csv_files = list(path.glob("*.csv"))
            file_path = csv_files[0]

        logger.info(f"Loading Synthetic Support dataset from {file_path}")

        conversations = self._load_csv(file_path)

        logger.info(
            f"Loaded {len(conversations)} conversations from Synthetic Support dataset"
        )

        return Dataset(
            name="SyntheticTechSupport",
            dataset_type=DatasetType.SYNTHETIC_SUPPORT,
            conversations=conversations,
            metadata={"source": "Synthetic Tech Support Chats", "file": str(file_path)},
        )

    def _load_csv(self, file_path: Path) -> List[Conversation]:
        """Load Synthetic Support CSV file."""
        conversations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for idx, row in enumerate(reader):
                    # Extract conversation data (adjust based on actual CSV structure)
                    # Assuming columns like: conversation_id, turn_number, speaker, text, etc.

                    # For now, treat each row as a single conversation
                    # This should be adjusted based on actual data structure
                    text = (
                        row.get("text", "")
                        or row.get("message", "")
                        or row.get("utterance", "")
                    )
                    speaker_str = row.get("speaker", "") or row.get("role", "user")

                    speaker = (
                        Speaker.ASSISTANT
                        if "agent" in speaker_str.lower()
                        or "assistant" in speaker_str.lower()
                        else Speaker.USER
                    )

                    turn = ConversationTurn(speaker=speaker, text=text)

                    conversation = Conversation(
                        id=f"synthetic_{idx}",
                        turns=[turn],
                        source_dataset=DatasetType.SYNTHETIC_SUPPORT,
                        metadata={"row_index": idx},
                    )

                    conversations.append(conversation)

        except Exception as e:
            logger.error(f"Error loading Synthetic Support CSV {file_path}: {e}")
            raise

        return conversations


class DatasetLoaderFactory:
    """Factory class for creating appropriate dataset loaders."""

    @staticmethod
    def get_loader(dataset_type: DatasetType) -> DatasetLoaderInterface:
        """Get the appropriate loader for a dataset type."""
        loaders = {
            DatasetType.BANKING77: Banking77Loader(),
            DatasetType.BITEXT: BitextLoader(),
            DatasetType.SCHEMA_GUIDED: SchemaGuidedLoader(),
            DatasetType.TWITTER_SUPPORT: TwitterSupportLoader(),
            DatasetType.SYNTHETIC_SUPPORT: SyntheticSupportLoader(),
        }

        loader = loaders.get(dataset_type)
        if not loader:
            raise ValueError(f"No loader available for dataset type: {dataset_type}")

        return loader

    @staticmethod
    def auto_detect_and_load(path: Path) -> Dataset:
        """Auto-detect dataset type and load it."""
        path = Path(path)

        # Try each loader
        loaders = [
            (DatasetType.BANKING77, Banking77Loader()),
            (DatasetType.BITEXT, BitextLoader()),
            (DatasetType.SCHEMA_GUIDED, SchemaGuidedLoader()),
            (DatasetType.TWITTER_SUPPORT, TwitterSupportLoader()),
            (DatasetType.SYNTHETIC_SUPPORT, SyntheticSupportLoader()),
        ]

        for dataset_type, loader in loaders:
            if loader.validate_format(path):
                logger.info(f"Auto-detected dataset type: {dataset_type.value}")
                return loader.load(path)

        raise ValueError(f"Could not auto-detect dataset type for path: {path}")
