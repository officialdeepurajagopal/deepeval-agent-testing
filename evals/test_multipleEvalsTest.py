import os
import sys
sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric, AnswerRelevancyMetric
from deepeval.tracing import observe, update_current_trace
from booking_agent import run_booking_agent as _run_booking_agent


@observe(name="booking_agent")
def run_booking_agent(user_input: str) -> str:
    return _run_booking_agent( user_input )

answer_relevancy = AnswerRelevancyMetric(threshold = 0.7, model = "gpt-4o")
step_efficiency = StepEfficiencyMetric(threshold=0.5,model = "gpt-4o")

prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=[
        "You are a friendly booking-support agent for AeroNova Airways. "
        "Keep replies short and helpful."
    ], threshold= 0.7, model = "gpt-4o"
)


dataset = EvaluationDataset(goldens = [

    Golden(input = "What's the status of my booking ANX7Q2?"),
    Golden( input="What is the baggage allowance for economy class?" ),
    Golden( input="How many bags can I check in on booking BKZ9M4?" )

])

for golden in dataset.evals_iterator(metrics = [prompt_alignment,step_efficiency,answer_relevancy]):
    run_booking_agent(golden.input)


    #sample output -
