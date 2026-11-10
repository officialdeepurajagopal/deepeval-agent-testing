"""Evaluate the instrumented support agent with DeepEval tracing."""

import os
import sys

# Allow running this file directly with `python tests/test_tracingComponents.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace

from booking_agent import run_booking_agent as _run_booking_agent


# Trace the agent and attach the expected tools from the active golden.
@observe(name="booking_agent")
def run_booking_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden and golden.expected_tools:
        update_current_trace(expected_tools=golden.expected_tools)

    return _run_booking_agent(user_input)


def build_dataset() -> EvaluationDataset:
    """Create the golden cases used by this evaluation."""
    return EvaluationDataset(
        goldens=[
            Golden(
                input="What's the status of my booking ANX7Q2?",
                expected_tools=[ToolCall(name="get_booking_status")],
            ),
            Golden(
                input="What is the baggage allowance for economy class?",
                expected_tools=[ToolCall(name="get_baggage_allowance")],
            ),
            # Routed to the MCP-served tool rather than a local one.
            Golden(
                input="Which flights do you have from London to New York?",
                expected_tools=[ToolCall(name="search_flights")],
            ),
        ]
    )


def build_metrics() -> list[TaskCompletionMetric | ToolCorrectnessMetric]:
    """Configure the judge and metrics for the evaluation."""
    judge = AnthropicModel(model="claude-sonnet-4-6")
    return [
        TaskCompletionMetric(threshold=0.7, model=judge),
        ToolCorrectnessMetric(threshold=0.7),
    ]


def main() -> None:
    dataset = build_dataset()
    metrics = build_metrics()

    for golden in dataset.evals_iterator(metrics=metrics):
        run_booking_agent(golden.input)


if __name__ == "__main__":
    main()
