#Contextual Precision -< Groundtruth (Expected output)
#50%
# 5 docs  - LLM(

#Recall - 10 docs-> Top 2 documents = 30% noise
import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
from deepeval.contextvars import get_current_golden
from deepeval.dataset import Golden
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, AnswerRelevancyMetric, \
    FaithfulnessMetric
from deepeval.tracing import observe, update_current_trace

from policy_rag_agent import run_policy_rag_agent as _run_policy_rag_agent

@observe(name="policy_rag_agent" )
def run_policy_rag_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_output:
            update_current_trace( expected_output=golden.expected_output )
    return _run_policy_rag_agent(user_input)


dataset = EvaluationDataset(goldens = [

    Golden(
        input = "Can I get a refund if I cancel my flight?",
        expected_output = (
            "Flexi fares can be cancelled up to 24 hours before departure for a full refund. "
            "Saver fares are non-refundable; refunds take 7–10 business days."
                 ),),

    Golden(
        input="How much does an extra checked bag cost?",
        expected_output=(
            "An extra checked bag costs $60 if added online at least 24 hours before departure, "
            "or $90 at the airport. Bags between 23 and 32 kg incur a $50 overweight fee."
        ),
    ),])

precisionMetric = ContextualPrecisionMetric(
    threshold=0.7,
    model = "gpt-4o"

)
recallMetric = ContextualRecallMetric(
    threshold=0.7,
    model="gpt-4o",
    include_reason=True
)

relevancyMetric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4o",
    include_reason=True
)


faithfulMetric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4o",
    include_reason=True
)





for golden in dataset.evals_iterator(metrics=[precisionMetric,recallMetric,relevancyMetric,faithfulMetric]):
    run_policy_rag_agent(golden.input)



