#Assertion - Judge LLM (Expected output) -> Actual output
import sys, os

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams
from deepeval.tracing import observe, update_current_trace
from booking_agent import run_booking_agent as _run_booking_agent


@observe(name="booking_agent")
def run_booking_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_output:
            update_current_trace( expected_output=golden.expected_output )

    return _run_booking_agent( user_input )


correctness = GEval(
    name = "Correctness",
    criteria=(
        "Determine whether the actual output conveys the same factual information "
        "as the expected output. Minor wording differences are acceptable; "
        "missing or wrong facts are not."
    ),
    model="gpt-4o",
    threshold=0.79,
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT  ])


dataset = EvaluationDataset(goldens = [
    Golden(input = "What's the status of my booking ANX7Q2?",
           expected_output= "Your booking ANX7Q2 is confirmed on flight AN 214 from Bengaluru (BLR) to Singapore (SIN) on 18 December 2026, departing at 08:40, in seat 14C.")
])

for golden in dataset.evals_iterator(metrics=[correctness]):
    run_booking_agent(golden.input)

















