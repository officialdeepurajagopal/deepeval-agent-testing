import sys, os

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

from deepeval.evaluate import evaluate
from deepeval.metrics import GEval, ConversationalGEval
from deepeval.test_case import Turn, ConversationalTestCase, MultiTurnParams

from support_chatbot import chat_turn

turns = []
history = []
for user_msg in [
        "Hi! I booked a flight last week, my booking reference is ANX7Q2.",

        "Is my flight still confirmed?",
        "What departure time did you just mention?",   # tests memory retention
        "Can I upgrade to business class?",
    ]:
       reply,history, _  = chat_turn(user_msg,history)
       turns.append(Turn(role="user",content=user_msg))
       turns.append(Turn(role="assistant",content=reply))

correctness = ConversationalGEval(
    name = "Correctness",
    criteria=(
         "Did the chatbot fully resolve the customer's issue? "
      "It should use tools when needed and provide accurate answers."
    ),
    model="gpt-4o",
    threshold=0.80,
    evaluation_params=[
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
         ])


test_case = ConversationalTestCase(
    turns = turns
)

evaluate(test_cases = [test_case], metrics = [correctness])