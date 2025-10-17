import time

from src.models.core import Dataset, DatasetType, Conversation, ConversationTurn, Speaker
from src.services.conversation_analyzer import ConversationFlowAnalyzer


def build_dataset(conversation_count: int = 200, turns_per_conversation: int = 6) -> Dataset:
    conversations = []
    for i in range(conversation_count):
        turns = []
        for j in range(turns_per_conversation):
            speaker = Speaker.USER if j % 2 == 0 else Speaker.ASSISTANT
            turns.append(
                ConversationTurn(
                    speaker=speaker,
                    text=f"Sample text {i}-{j}",
                    intent="intent_a" if speaker == Speaker.USER else None,
                )
            )
        conversations.append(
            Conversation(
                id=f"conv-{i}",
                turns=turns,
                source_dataset=DatasetType.BANKING77,
                success=True,
            )
        )
    return Dataset(
        name="PerformanceDataset",
        dataset_type=DatasetType.BANKING77,
        conversations=conversations,
    )


def test_conversation_flow_performance():
    dataset = build_dataset()
    analyzer = ConversationFlowAnalyzer()
    start = time.perf_counter()
    result = analyzer.analyze_dialogue_flow(dataset.conversations)
    duration = (time.perf_counter() - start) * 1000
    assert result["conversations_analyzed"] == len(dataset.conversations)
    assert duration < 500.0  # should run well under half a second
