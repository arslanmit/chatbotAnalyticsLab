from datetime import datetime

from src.models.core import Dataset, DatasetType, Conversation, ConversationTurn, Speaker


def sample_dataset(name: str = "TestDataset") -> Dataset:
    conversation = Conversation(
        id="conv-1",
        turns=[
            ConversationTurn(
                speaker=Speaker.USER,
                text="Hello",
                timestamp=datetime.utcnow(),
                intent="greet",
                confidence=0.9,
            ),
            ConversationTurn(
                speaker=Speaker.ASSISTANT,
                text="Hi there!",
                timestamp=datetime.utcnow(),
            ),
        ],
        source_dataset=DatasetType.BANKING77,
        metadata={"topic": "greeting"},
        success=True,
    )
    return Dataset(
        name=name,
        dataset_type=DatasetType.BANKING77,
        conversations=[conversation],
        metadata={"path": "/tmp/test"},
    )
