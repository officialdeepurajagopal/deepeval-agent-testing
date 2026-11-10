from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric
from deepeval.models import AnthropicModel
from deepeval.tracing import observe, update_current_trace
from policy_rag_agent import run_policy_rag_agent as _run_policy_rag_agent

@observe(name="policy_rag_agent")
def run_policy_rag_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return _run_policy_rag_agent(user_input)


def build_dataset() -> EvaluationDataset:
    """Create the golden cases used by this evaluation."""
    return EvaluationDataset(
        goldens=[
            Golden(
                input="Can I get a refund if I cancel my flight?",
                expected_output=(
                    "Flexi fares can be cancelled up to 24 hours before departure for a full refund. "
                    "Saver fares are non-refundable, but taxes and airport fees are refunded on request. "
                    "Refunds are returned to the original payment method within 7–10 business days."
                )
            ),
            Golden(
                input="How early do I need to check in for an international flight?",
                expected_output=(
                    "Airport check-in counters close 75 minutes before international flights. "
                    "Online check-in closes 90 minutes before departure."
                )
            )
        ]
    )

judge = AnthropicModel(model="claude-sonnet-4-6")

def build_metrics() -> list[ContextualPrecisionMetric]:
    """Configure the judge and metrics for the evaluation."""
    return [
        ContextualPrecisionMetric(
            threshold=0.7,
            model=judge,
        ),
        ContextualRecallMetric(
            threshold=0.7,
            model=judge,
        ),
        FaithfulnessMetric(
            threshold=0.7,
            model=judge,
        )
    ]


dataset = build_dataset()
metrics = build_metrics()

for golden in dataset.evals_iterator(metrics=metrics):
    run_policy_rag_agent(golden.input)
