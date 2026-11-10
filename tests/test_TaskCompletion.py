import os
import sys

# Allow running this file directly with `python tests/test_TaskCompletion.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric
from deepeval.models import AnthropicModel
from deepeval.tracing import observe

from booking_agent import run_booking_agent as _run_booking_agent


# Wrap the agent in @observe so DeepEval creates a top-level trace.
# The agent's CallbackHandler attaches LLM + tool spans to it, so the
# TaskCompletion judge can see which tools were actually called.
@observe(name="booking_agent")
def run_booking_agent(user_input: str) -> str:
    return _run_booking_agent(user_input)


judge = AnthropicModel(model="claude-sonnet-4-6")
task_completion = TaskCompletionMetric(threshold=0.7, model=judge)

dataset = EvaluationDataset(goldens=[
    Golden(input="What's the status of my booking ANX7Q2?"),
])

for golden in dataset.evals_iterator(metrics=[task_completion]):
    run_booking_agent(golden.input)
