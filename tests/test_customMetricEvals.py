"""Evaluate the support agent's correctness against expected outputs with GEval."""

import os
import sys

# Allow running this file directly with `python tests/test_customMetricEvals.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.models import AnthropicModel
from deepeval.test_case import SingleTurnParams
from deepeval.tracing import observe, update_current_trace

from booking_agent import run_booking_agent as _run_booking_agent


# Trace the agent and attach the expected output from the active golden,
# since GEval reads expected_output from the trace, not the golden.
@observe(name="booking_agent")
def run_booking_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden and golden.expected_output:
        update_current_trace(expected_output=golden.expected_output)

    return _run_booking_agent(user_input)


def build_dataset() -> EvaluationDataset:
    """Create the golden cases used by this evaluation."""
    return EvaluationDataset(
        goldens=[
            Golden(
                input="What's the status of my booking ANX7Q2?",
                expected_output=(
                    "Your booking ANX7Q2 is confirmed on flight AN 214 from Bengaluru (BLR) to Singapore (SIN) on 18 December 2026, departing at 08:40, in seat 14C."
                ),
            )
        ]
    )


judge = AnthropicModel(model="claude-sonnet-4-6")

correctness = GEval(
    name="Correctness",
    criteria=(
        "Determine if the agent's response is factually accurate and addresses the "
        "user's query appropriately. It should match the expected output. Minor "
        "wording differences are acceptable as long as the core information is "
        "correct. Missing or wrong facts should be marked as incorrect."
    ),
    model=judge,
    threshold=0.8,
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
)


def build_metrics() -> list[GEval]:
    """Return the metrics for the evaluation."""
    return [correctness]


def main() -> None:
    dataset = build_dataset()
    metrics = build_metrics()

    for golden in dataset.evals_iterator(metrics=metrics):
        run_booking_agent(golden.input)


if __name__ == "__main__":
    main()
