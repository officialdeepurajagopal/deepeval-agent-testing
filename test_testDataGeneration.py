import os.path

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import TaskCompletionMetric, BiasMetric, PIILeakageMetric, ToxicityMetric
from deepeval.models import AnthropicModel
from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.tracing import observe, update_current_trace

from booking_agent import run_booking_agent as _run_booking_agent


@observe(name="booking_agent")
def run_booking_agent(user_input: str) -> str:
    if golden:
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return _run_booking_agent(user_input)


judge = AnthropicModel(model="claude-sonnet-4-6")


def load_contexts(path: str) -> list[list[str]]:
    """Split policies.txt into one context per section (sections are blank-line separated).

    generate_goldens_from_docs() needs an embedding model to chunk documents, and
    DeepEval's default embedder is OpenAI. Anthropic has no embeddings API, so we
    build the contexts ourselves and use generate_goldens_from_contexts() instead.
    """
    with open(path) as f:
        sections = [s.strip() for s in f.read().split("\n\n") if s.strip()]
    return [[section] for section in sections[1:]]   # skip the document title


synthesizer = Synthesizer(model=judge)
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=load_contexts(os.path.join(os.path.dirname(__file__), "policies.txt")),
    include_expected_output=True,
    max_goldens_per_context=2,
)

dataset = EvaluationDataset(goldens=goldens)

# Synthesized goldens have no expected_tools, so ToolCorrectnessMetric can't score
# them. TaskCompletion judges the agent's trace (including tool calls) instead.
metric = [BiasMetric(threshold=0.5, model=judge), PIILeakageMetric(threshold=0.5, model=judge), ToxicityMetric(threshold=0.5, model=judge)]

for golden in dataset.evals_iterator(metrics=metric):
    run_booking_agent(golden.input)
