"""
Performance analytics service aggregating metrics across predictions and conversations.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from src.interfaces.base import PerformanceAnalyzerInterface
from src.models.core import Conversation, ConversationTurn, IntentPrediction, Speaker
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceAnalyzer(PerformanceAnalyzerInterface):
    """Calculate high-level performance metrics and generate actionable reports."""

    def calculate_intent_distribution(self, predictions: List[IntentPrediction]) -> Dict[str, int]:
        """Count occurrences of intents across predictions."""
        distribution = Counter()
        for prediction in predictions:
            distribution[prediction.intent] += 1
        logger.debug("Computed intent distribution for %d predictions", len(predictions))
        return dict(distribution)

    def compute_response_times(self, conversations: List[Conversation]) -> Dict[str, float]:
        """Compute response time statistics based on turn timestamps."""
        if not conversations:
            return {}

        response_times: List[float] = []
        per_conversation_avg: List[float] = []
        missing_timestamps = 0

        for conversation in conversations:
            conversation_deltas: List[float] = []
            for user_turn, assistant_turn in self._iterate_user_assistant_pairs(conversation.turns):
                if user_turn.timestamp and assistant_turn.timestamp:
                    delta = (assistant_turn.timestamp - user_turn.timestamp).total_seconds()
                    if delta >= 0:
                        response_times.append(delta)
                        conversation_deltas.append(delta)
                else:
                    missing_timestamps += 1
            if conversation_deltas:
                per_conversation_avg.append(mean(conversation_deltas))

        if not response_times:
            logger.warning("Unable to compute response times: no valid timestamp pairs found")
            return {"missing_pairs": missing_timestamps}

        stats = {
            "average_response_time": round(mean(response_times), 2),
            "median_response_time": round(median(response_times), 2),
            "p90_response_time": round(self._percentile(response_times, 90), 2),
            "conversation_average_response_time": round(mean(per_conversation_avg), 2)
            if per_conversation_avg
            else 0.0,
            "samples": len(response_times),
            "missing_pairs": missing_timestamps,
        }

        logger.info(
            "Computed response time stats (avg=%.2fs, median=%.2fs, p90=%.2fs)",
            stats["average_response_time"],
            stats["median_response_time"],
            stats["p90_response_time"],
        )
        return stats

    def generate_performance_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine supplied metrics into a high-level report and surface key alerts.

        Expected keys in `metrics`:
            - success_metrics: Dict[str, float]
            - response_times: Dict[str, float]
            - intent_distribution: Dict[str, int]
            - sentiment: Dict[str, Any]
        """
        report_time = datetime.utcnow().isoformat()
        alerts: List[str] = []

        success_metrics = metrics.get("success_metrics", {})
        response_times = metrics.get("response_times", {})
        sentiment = metrics.get("sentiment", {})

        if success_metrics:
            success_rate = success_metrics.get("success_rate", 0.0)
            failure_rate = success_metrics.get("failure_rate", 0.0)
            if success_rate < 0.8:
                alerts.append(
                    f"Success rate is below target at {success_rate * 100:.1f}%"
                )
            if failure_rate > 0.15:
                alerts.append(
                    f"Failure rate is elevated at {failure_rate * 100:.1f}%"
                )

        if response_times:
            average_rt = response_times.get("average_response_time", 0.0)
            p90_rt = response_times.get("p90_response_time", 0.0)
            if average_rt > 45:
                alerts.append(
                    f"Average response time ({average_rt:.1f}s) exceeds the 45s threshold"
                )
            if p90_rt > 120:
                alerts.append(
                    f"90th percentile response time ({p90_rt:.1f}s) indicates slow resolutions"
                )

        if sentiment:
            aggregate = sentiment.get("aggregate", {})
            sentiment_gap = aggregate.get("overall_sentiment_gap", 0.0)
            if sentiment_gap < -0.2:
                alerts.append(
                    "Customer sentiment is trending negatively relative to assistant tone"
                )

        report = {
            "generated_at": report_time,
            "success_metrics": success_metrics,
            "response_times": response_times,
            "intent_distribution": metrics.get("intent_distribution", {}),
            "sentiment_summary": sentiment.get("aggregate"),
            "alerts": alerts,
        }

        logger.info(
            "Generated performance report with %d alerts", len(alerts)
        )
        return report

    # Internal helpers ----------------------------------------------------------

    @staticmethod
    def _iterate_user_assistant_pairs(
        turns: List[ConversationTurn],
    ) -> List[Tuple[ConversationTurn, ConversationTurn]]:
        pairs: List[Tuple[ConversationTurn, ConversationTurn]] = []
        previous_user_turn: Optional[ConversationTurn] = None
        for turn in turns:
            if turn.speaker == Speaker.USER:
                previous_user_turn = turn
            elif turn.speaker == Speaker.ASSISTANT and previous_user_turn:
                pairs.append((previous_user_turn, turn))
                previous_user_turn = None
        return pairs

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (percentile / 100)
        f = int(k)
        c = f + 1
        if c >= len(sorted_values):
            return sorted_values[f]
        d0 = sorted_values[f] * (c - k)
        d1 = sorted_values[c] * (k - f)
        return d0 + d1
