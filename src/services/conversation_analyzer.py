"""
Conversation flow analysis service providing dialogue sequencing, state tracking,
and failure point detection across conversations.
"""

from collections import Counter
from statistics import mean, median
from typing import Dict, Any, List, Optional, Set

from src.interfaces.base import ConversationAnalyzerInterface
from src.models.core import Conversation, ConversationTurn, Speaker
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ConversationFlowAnalyzer(ConversationAnalyzerInterface):
    """
    Analyze conversations to understand dialogue flow, detect failure points,
    and compute success metrics.
    """

    def __init__(
        self,
        greeting_keywords: Optional[List[str]] = None,
        closing_keywords: Optional[List[str]] = None,
        frustration_keywords: Optional[List[str]] = None,
        fallback_phrases: Optional[List[str]] = None,
    ):
        self.greeting_keywords = greeting_keywords or [
            "hello",
            "hi ",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
            "hey",
        ]
        self.closing_keywords = closing_keywords or [
            "thank you",
            "thanks",
            "appreciate",
            "that helps",
            "resolved",
            "great",
            "perfect",
            "bye",
            "goodbye",
        ]
        self.frustration_keywords = frustration_keywords or [
            "not helpful",
            "this is useless",
            "frustrated",
            "angry",
            "unhelpful",
            "upset",
            "complain",
            "cancel",
            "make a complaint",
            "speak to a human",
            "agent",
            "representative",
        ]
        self.fallback_phrases = fallback_phrases or [
            "i'm sorry i didn't understand",
            "could you please rephrase",
            "i'm still learning",
            "i do not understand",
            "i'm not sure",
            "cannot help with that",
        ]

    # ConversationAnalyzerInterface implementation ---------------------------------

    def analyze_dialogue_flow(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Analyze overall dialogue flow across conversations."""

        if not conversations:
            logger.warning("No conversations provided for flow analysis")
            return {
                "conversations_analyzed": 0,
                "turn_statistics": {},
                "speaker_turn_ratio": {},
                "state_distribution": {},
                "transition_matrix": {},
                "state_transitions": {},
                "flow_details": [],
            }

        turn_counts: List[int] = []
        user_turns: int = 0
        assistant_turns: int = 0
        state_counter: Counter = Counter()
        speaker_transitions: Counter = Counter()
        state_transitions: Counter = Counter()
        flow_details: List[Dict[str, Any]] = []

        for conversation in conversations:
            turn_counts.append(conversation.turn_count)
            conversation_flow = {
                "conversation_id": conversation.id,
                "turn_count": conversation.turn_count,
                "turns": [],
            }

            previous_speaker: Optional[str] = None
            previous_state: Optional[str] = None

            for index, turn in enumerate(conversation.turns):
                state = self._categorize_turn(turn)
                state_counter[state] += 1

                if turn.speaker == Speaker.USER:
                    user_turns += 1
                else:
                    assistant_turns += 1

                flow_record = {
                    "index": index,
                    "speaker": turn.speaker.value,
                    "state": state,
                    "intent": turn.intent,
                    "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
                    "text_length": len(turn.text.split()) if turn.text else 0,
                }
                conversation_flow["turns"].append(flow_record)

                if previous_speaker:
                    transition_key = f"{previous_speaker}->{turn.speaker.value}"
                    speaker_transitions[transition_key] += 1
                if previous_state:
                    state_transition_key = f"{previous_state}->{state}"
                    state_transitions[state_transition_key] += 1

                previous_speaker = turn.speaker.value
                previous_state = state

            flow_details.append(conversation_flow)

        turn_stats = self._calculate_turn_statistics(turn_counts)
        speaker_ratio = self._calculate_speaker_ratio(user_turns, assistant_turns)

        logger.info(
            "Analyzed dialogue flow for %d conversations (avg_turns=%.2f)",
            len(conversations),
            turn_stats.get("average_turns", 0.0),
        )

        return {
            "conversations_analyzed": len(conversations),
            "turn_statistics": turn_stats,
            "speaker_turn_ratio": speaker_ratio,
            "state_distribution": dict(state_counter),
            "transition_matrix": dict(speaker_transitions),
            "state_transitions": dict(state_transitions),
            "flow_details": flow_details,
        }

    def detect_failure_points(self, conversations: List[Conversation]) -> List[Dict[str, Any]]:
        """Detect potential failure points within conversations."""

        failure_points: List[Dict[str, Any]] = []

        for conversation in conversations:
            conversation_failures = []

            # Flag conversations explicitly marked unsuccessful
            if conversation.success is False:
                conversation_failures.append(
                    {
                        "conversation_id": conversation.id,
                        "turn_index": None,
                        "speaker": None,
                        "reason": "conversation_marked_unsuccessful",
                        "confidence": 1.0,
                    }
                )

            for index, turn in enumerate(conversation.turns):
                text = (turn.text or "").lower()

                if not text:
                    continue

                if turn.speaker == Speaker.ASSISTANT and self._contains_any(text, self.fallback_phrases):
                    conversation_failures.append(
                        {
                            "conversation_id": conversation.id,
                            "turn_index": index,
                            "speaker": turn.speaker.value,
                            "reason": "assistant_fallback_response",
                            "confidence": 0.7,
                        }
                    )

                if turn.speaker == Speaker.USER and self._contains_any(text, self.frustration_keywords):
                    conversation_failures.append(
                        {
                            "conversation_id": conversation.id,
                            "turn_index": index,
                            "speaker": turn.speaker.value,
                            "reason": "user_frustration_detected",
                            "confidence": 0.6,
                        }
                    )

            # Detect unresolved questions (conversation ends with user question)
            if conversation.turns:
                last_turn = conversation.turns[-1]
                if last_turn.speaker == Speaker.USER and self._is_question(last_turn.text or ""):
                    conversation_failures.append(
                        {
                            "conversation_id": conversation.id,
                            "turn_index": len(conversation.turns) - 1,
                            "speaker": last_turn.speaker.value,
                            "reason": "conversation_ended_with_unanswered_question",
                            "confidence": 0.5,
                        }
                    )

            failure_points.extend(conversation_failures)

        logger.debug("Detected %d failure points across conversations", len(failure_points))
        return failure_points

    def calculate_success_metrics(self, conversations: List[Conversation]) -> Dict[str, float]:
        """Calculate conversation success metrics."""

        if not conversations:
            logger.warning("No conversations provided for success metrics calculation")
            return {}

        known_successes = [conv.success for conv in conversations if conv.success is not None]
        failure_points = self.detect_failure_points(conversations)
        failed_conversation_ids = {fp["conversation_id"] for fp in failure_points}

        if known_successes:
            success_rate = sum(1 for flag in known_successes if flag) / len(known_successes)
        else:
            derived_success = len(conversations) - len(failed_conversation_ids)
            success_rate = derived_success / len(conversations)

        resolution_turns: List[int] = []
        frustration_conversations: Set[str] = set()

        for conversation in conversations:
            for index, turn in enumerate(conversation.turns):
                state = self._categorize_turn(turn)
                if state in {"resolution", "closing"} and turn.speaker == Speaker.ASSISTANT:
                    resolution_turns.append(index + 1)
                    break
                if turn.speaker == Speaker.USER and state == "frustration":
                    frustration_conversations.add(conversation.id)

        average_resolution_turn = mean(resolution_turns) if resolution_turns else 0.0
        failure_rate = len(failed_conversation_ids) / len(conversations)
        frustration_rate = len(frustration_conversations) / len(conversations)

        metrics = {
            "success_rate": round(success_rate, 4),
            "failure_rate": round(failure_rate, 4),
            "average_resolution_turn": round(average_resolution_turn, 2),
            "customer_frustration_rate": round(frustration_rate, 4),
        }

        logger.info(
            "Calculated conversation success metrics (success_rate=%.2f%%, failure_rate=%.2f%%)",
            metrics["success_rate"] * 100,
            metrics["failure_rate"] * 100,
        )
        return metrics

    # Internal helpers -----------------------------------------------------------

    @staticmethod
    def _calculate_turn_statistics(turn_counts: List[int]) -> Dict[str, float]:
        if not turn_counts:
            return {}
        return {
            "average_turns": round(mean(turn_counts), 2),
            "median_turns": float(median(turn_counts)),
            "min_turns": float(min(turn_counts)),
            "max_turns": float(max(turn_counts)),
        }

    @staticmethod
    def _calculate_speaker_ratio(user_turns: int, assistant_turns: int) -> Dict[str, float]:
        total_turns = user_turns + assistant_turns
        if total_turns == 0:
            return {"user": 0.0, "assistant": 0.0}

        return {
            "user": round(user_turns / total_turns, 4),
            "assistant": round(assistant_turns / total_turns, 4),
        }

    def _categorize_turn(self, turn: ConversationTurn) -> str:
        text = (turn.text or "").lower()
        if not text:
            return "empty"

        if self._contains_any(text, self.greeting_keywords):
            return "greeting"

        if self._contains_any(text, self.closing_keywords):
            return "closing"

        if turn.speaker == Speaker.USER:
            if self._contains_any(text, self.frustration_keywords):
                return "frustration"
            if self._is_question(text):
                return "question"
            return "statement"

        if turn.speaker == Speaker.ASSISTANT:
            if self._contains_any(text, self.fallback_phrases):
                return "fallback"
            if any(phrase in text for phrase in ("let me check", "looking into", "investigating")):
                return "clarification"
            if any(phrase in text for phrase in ("i can help", "here's what", "steps you can")):
                return "resolution"
            return "response"

        return "unknown"

    @staticmethod
    def _contains_any(text: str, phrases: List[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _is_question(text: str) -> bool:
        question_words = (
            "how",
            "what",
            "why",
            "when",
            "where",
            "who",
            "can",
            "could",
            "would",
            "is ",
            "are ",
            "do ",
            "does ",
        )
        text = text.strip()
        if not text:
            return False
        return text.endswith("?") or text.startswith(question_words)
