"""Evaluate the support agent's adherence to its system prompt."""

import os
import sys

# Allow running this file directly with `python tests/test_multipleEvalsTest.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric, AnswerRelevancyMetric
from deepeval.models import AnthropicModel
from deepeval.tracing import observe

from booking_agent import run_booking_agent as _run_booking_agent


@observe(name="booking_agent")
def run_booking_agent(user_input: str) -> str:
    return _run_booking_agent(user_input)


def build_dataset() -> EvaluationDataset:
    """Create the golden cases used by this evaluation."""
    return EvaluationDataset(
        goldens=[
            Golden(input="What's the status of my booking ANX7Q2?"),
            Golden(input="What is the baggage allowance for economy class?"),
            Golden(input="How many bags can I check in on booking BKZ9M4?"),
        ]
    )


def build_metrics() -> list[PromptAlignmentMetric]:
    """Configure the judge and metrics for the evaluation."""
    judge = AnthropicModel(model="claude-sonnet-4-6")
    return [
        PromptAlignmentMetric(
            prompt_instructions=[
                "You are a friendly booking-support agent for AeroNova Airways. "
                "Use the available tools to answer booking status, baggage and flight search questions. "
                "Keep replies short and helpful."
            ],
            threshold=0.7,
            model=judge,
        ),
        StepEfficiencyMetric(threshold=0.5, model=judge),
        AnswerRelevancyMetric(threshold=0.7, model=judge)
    ]


def main():
    dataset = build_dataset()
    metrics = build_metrics()

    for golden in dataset.evals_iterator(metrics=metrics):
        run_booking_agent(golden.input)


if __name__ == "__main__":
    main()
