"""
Rule-based sentiment analysis utilities tailored for customer support conversations.

Provides per-turn sentiment scoring, trend analysis over time, and satisfaction
metrics that correlate sentiment signals with conversation outcomes.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, Any, List, Optional

from src.models.core import Conversation, ConversationTurn, Speaker
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Compute sentiment insights for chatbot conversations using a lightweight lexicon."""

    def __init__(
        self,
        positive_lexicon: Optional[Dict[str, float]] = None,
        negative_lexicon: Optional[Dict[str, float]] = None,
    ):
        self.positive_lexicon = positive_lexicon or {
            "good": 1.0,
            "great": 1.2,
            "excellent": 1.5,
            "awesome": 1.3,
            "helpful": 1.0,
            "thanks": 0.8,
            "thank": 0.8,
            "appreciate": 1.0,
            "love": 1.4,
            "satisfied": 1.3,
            "happy": 1.2,
            "perfect": 1.4,
            "resolved": 1.1,
            "fantastic": 1.5,
            "success": 1.0,
        }
        self.negative_lexicon = negative_lexicon or {
            "bad": 1.0,
            "terrible": 1.6,
            "awful": 1.5,
            "poor": 1.0,
            "worse": 1.2,
            "worst": 1.6,
            "frustrated": 1.4,
            "angry": 1.4,
            "upset": 1.2,
            "unhelpful": 1.3,
            "issue": 0.8,
            "problem": 1.0,
            "complain": 1.3,
            "cancel": 1.0,
            "hate": 1.5,
            "annoyed": 1.3,
            "disappointed": 1.4,
        }
        self.negation_tokens = {"not", "never", "no", "cannot", "can't", "dont", "don't"}
        self.intensifiers = {"really", "very", "extremely", "super", "highly", "so"}

    # Public analysis helpers ---------------------------------------------------

    def score_text(self, text: str) -> float:
        """Score raw text and return a value in the range [-1.0, 1.0]."""
        if not text:
            return 0.0

        tokens = [token.strip(".,!?") for token in text.lower().split()]
        score = 0.0
        sentiment_hits = 0
        negation_window = 0
        intensity_multiplier = 1.0

        for token in tokens:
            if not token:
                continue

            if token in self.intensifiers:
                intensity_multiplier = 1.5
                continue

            if token in self.negation_tokens:
                negation_window = 3
                continue

            token_score = 0.0
            if token in self.positive_lexicon:
                token_score = self.positive_lexicon[token]
            elif token in self.negative_lexicon:
                token_score = -self.negative_lexicon[token]

            if token_score != 0.0:
                sentiment_hits += 1
                if negation_window > 0:
                    token_score *= -1
                    negation_window -= 1
                score += token_score * intensity_multiplier
                intensity_multiplier = 1.0
            else:
                if negation_window > 0:
                    negation_window -= 1
                intensity_multiplier = 1.0

        if sentiment_hits == 0:
            return 0.0

        normalized = score / max(sentiment_hits, 1)
        bounded = max(-1.0, min(1.0, normalized))
        return round(bounded, 4)

    def analyze_conversations(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Compute sentiment summaries for each conversation and roll up global metrics."""
        if not conversations:
            logger.warning("No conversations provided for sentiment analysis")
            return {"conversations_analyzed": 0, "conversation_sentiment": [], "aggregate": {}}

        conversation_results: List[Dict[str, Any]] = []
        all_user_scores: List[float] = []
        all_assistant_scores: List[float] = []

        for conversation in conversations:
            turn_scores: List[float] = []
            user_scores: List[float] = []
            assistant_scores: List[float] = []

            for turn in conversation.turns:
                score = self.score_text(turn.text)
                turn_scores.append(score)
                if turn.speaker == Speaker.USER:
                    user_scores.append(score)
                    all_user_scores.append(score)
                elif turn.speaker == Speaker.ASSISTANT:
                    assistant_scores.append(score)
                    all_assistant_scores.append(score)

            conversation_results.append(
                {
                    "conversation_id": conversation.id,
                    "average_sentiment": round(mean(turn_scores), 4) if turn_scores else 0.0,
                    "user_sentiment": round(mean(user_scores), 4) if user_scores else 0.0,
                    "assistant_sentiment": round(mean(assistant_scores), 4) if assistant_scores else 0.0,
                    "start_sentiment": turn_scores[0] if turn_scores else 0.0,
                    "end_sentiment": turn_scores[-1] if turn_scores else 0.0,
                }
            )

        aggregate = {
            "overall_average_user_sentiment": round(mean(all_user_scores), 4) if all_user_scores else 0.0,
            "overall_average_assistant_sentiment": round(mean(all_assistant_scores), 4) if all_assistant_scores else 0.0,
            "overall_sentiment_gap": round(
                (
                    (mean(all_user_scores) if all_user_scores else 0.0)
                    - (mean(all_assistant_scores) if all_assistant_scores else 0.0)
                ),
                4,
            ),
        }

        logger.info(
            "Calculated sentiment for %d conversations (avg_user=%.3f, avg_assistant=%.3f)",
            len(conversations),
            aggregate["overall_average_user_sentiment"],
            aggregate["overall_average_assistant_sentiment"],
        )

        return {
            "conversations_analyzed": len(conversations),
            "conversation_sentiment": conversation_results,
            "aggregate": aggregate,
        }

    def calculate_sentiment_trend(
        self, conversations: List[Conversation], granularity: str = "daily"
    ) -> Dict[str, Any]:
        """
        Build sentiment trends over time for customer turns.

        Args:
            conversations: Conversations to analyze
            granularity: 'daily', 'hourly', or 'conversation'
        """
        if not conversations:
            return {"granularity": granularity, "trend": []}

        buckets: Dict[str, List[float]] = defaultdict(list)

        for conversation in conversations:
            for turn in conversation.turns:
                if turn.speaker != Speaker.USER:
                    continue
                score = self.score_text(turn.text)
                if turn.timestamp is None or granularity == "conversation":
                    key = f"{conversation.id}"
                else:
                    if granularity == "hourly":
                        key = turn.timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
                    else:  # daily by default
                        key = turn.timestamp.date().isoformat()
                buckets[key].append(score)

        trend = [
            {
                "bucket": bucket,
                "average_sentiment": round(mean(scores), 4) if scores else 0.0,
                "samples": len(scores),
            }
            for bucket, scores in sorted(buckets.items())
        ]

        return {"granularity": granularity, "trend": trend}

    def calculate_satisfaction_metrics(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Estimate customer satisfaction based on sentiment shifts and closing signals."""
        if not conversations:
            return {"conversations_evaluated": 0}

        conversation_count = 0
        satisfied_count = 0
        detractor_count = 0
        neutral_count = 0
        satisfaction_bonus = 0.0
        deltas: List[float] = []

        for conversation in conversations:
            user_turns = [turn for turn in conversation.turns if turn.speaker == Speaker.USER]
            if not user_turns:
                continue

            conversation_count += 1
            start_score = self.score_text(user_turns[0].text)
            end_score = self.score_text(user_turns[-1].text)
            delta = end_score - start_score
            deltas.append(delta)

            if end_score >= 0.2:
                satisfied_count += 1
            elif end_score <= -0.2:
                detractor_count += 1
            else:
                neutral_count += 1

            # Boost satisfaction for successful conversations
            if conversation.success:
                satisfaction_bonus += 0.25

        if conversation_count == 0:
            return {"conversations_evaluated": 0}

        satisfaction_index = (
            (satisfied_count - detractor_count + satisfaction_bonus) / conversation_count
        )

        metrics = {
            "conversations_evaluated": conversation_count,
            "satisfied_ratio": round(satisfied_count / conversation_count, 4),
            "detractor_ratio": round(detractor_count / conversation_count, 4),
            "neutral_ratio": round(neutral_count / conversation_count, 4),
            "average_sentiment_delta": round(mean(deltas), 4) if deltas else 0.0,
            "satisfaction_index": round(satisfaction_index, 4),
        }

        logger.info(
            "Computed satisfaction metrics (index=%.3f, satisfied_ratio=%.3f)",
            metrics["satisfaction_index"],
            metrics["satisfied_ratio"],
        )
        return metrics
