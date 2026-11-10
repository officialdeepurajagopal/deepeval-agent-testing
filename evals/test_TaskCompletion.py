import os
import sys

from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase

from booking_agent import run_booking_agent
sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

actual_output = run_booking_agent("What's the status of my booking ANX7Q2?")

test_case = LLMTestCase(
    input = "What's the status of my booking ANX7Q2?",
    actual_output = actual_output
)

evaluate(test_cases = [test_case],
         metrics= [TaskCompletionMetric(threshold=0.7,model = "gpt-4o")])