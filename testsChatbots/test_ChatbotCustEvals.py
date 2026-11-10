"""Evaluate a multi-turn conversation with the support chatbot."""

import os
import sys

# Allow running this file directly with `python testsChatbots/test_ChatbotCustEvals.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deepeval.evaluate import evaluate
from deepeval.metrics import ConversationalGEval
from deepeval.models import AnthropicModel
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, ToolCall, Turn

from support_chatbot import chat_turn

judge = AnthropicModel(model="claude-sonnet-4-6")

# GEval only accepts single-turn LLMTestCase; ConversationalGEval is the
# multi-turn equivalent for ConversationalTestCase.
correctness = ConversationalGEval(
    name="Correctness",
    criteria=(
        "Did the chatbot provide a factually accurate and relevant response to the user's query? "
        "It should use the tools when needed and provide accurate answers. Minor wording differences are acceptable as long as the core information is correct. "
        "Missing or wrong facts should be marked as incorrect."
    ),
    model=judge,
    threshold=0.8,
    evaluation_params=[
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
        MultiTurnParams.TOOLS_CALLED,
    ],
)

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
        reply, history, tools = chat_turn(user_msg, history)
        turns.append(Turn(role="user", content=user_msg))
        turns.append(Turn(
            role="assistant",
            content=reply,
            tools_called=[
                ToolCall(name=t["name"], input_parameters=t["args"], output=t["result"])
                for t in tools
            ] or None,
        ))

    return ConversationalTestCase(turns=turns)


def build_metrics() -> list[ConversationalGEval]:
    """Configure the judge and metrics for the evaluation."""
    return [correctness]




def main() -> None:
    test_case = run_conversation(USER_MESSAGES)
    evaluate(test_cases=[test_case], metrics=build_metrics())


if __name__ == "__main__":
    main()
