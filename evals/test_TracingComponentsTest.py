import sys, os

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace

from booking_agent import run_booking_agent as _run_booking_agent

#Trace have expected values and actual values
@observe(name="booking_agent")
def run_booking_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace( expected_tools=golden.expected_tools )
        if golden.expected_output:
            update_current_trace( expected_output=golden.expected_output )

    return _run_booking_agent( user_input )




task_completion = TaskCompletionMetric(threshold=0.7,model="gpt-4o")

tool_correctness = ToolCorrectnessMetric()

dataSet = EvaluationDataset(goldens= [
    Golden(input = "What's the status of my booking ANX7Q2?",
           expected_tools=[ToolCall(name="get_booking_status")]),

    Golden(input = "What is the baggage allowance for economy class?",
           expected_tools=[ToolCall(name="get_baggage_allowance")])
])


for golden in dataSet.evals_iterator(metrics =[task_completion,tool_correctness]):
    run_booking_agent(golden.input)

