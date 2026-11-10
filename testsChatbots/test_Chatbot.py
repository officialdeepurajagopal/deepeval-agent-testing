"""Evaluate a multi-turn conversation with the support chatbot."""

import os
import sys

# Allow running this file directly with `python testsChatbots/test_Chatbot.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deepeval.evaluate import evaluate
from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric, ConversationCompletenessMetric, GEval
from deepeval.models import AnthropicModel
from deepeval.test_case import ConversationalTestCase, Turn

from support_chatbot import chat_turn


USER_MESSAGES = [
    "What's the status of my booking ANX7Q2?",
    "What is the baggage allowance for economy class?",
    "How much checked baggage can I take on that booking?",
]


def run_conversation(user_messages: list[str]) -> ConversationalTestCase:
    """Drive the chatbot live and record each user/assistant turn."""
    history: list[dict] = []
    turns: list[Turn] = []

    for user_msg in user_messages:
        reply, history, _ = chat_turn(user_msg, history)
        turns.append(Turn(role="user", content=user_msg))
        turns.append(Turn(role="assistant", content=reply))

    return ConversationalTestCase(turns=turns)


def build_metrics() -> list[TurnRelevancyMetric]:
    """Configure the judge and metrics for the evaluation."""
    judge = AnthropicModel(model="claude-sonnet-4-6")
    return [TurnRelevancyMetric(threshold=0.7, model=judge),
            KnowledgeRetentionMetric(threshold=0.8, model=judge),
            ConversationCompletenessMetric(threshold=0.8, model=judge)]


def main() -> None:
    test_case = run_conversation(USER_MESSAGES)
    evaluate(test_cases=[test_case], metrics=build_metrics())


if __name__ == "__main__":
    main()
