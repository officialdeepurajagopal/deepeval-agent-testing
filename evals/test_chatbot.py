import sys, os

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

from deepeval.evaluate import evaluate
from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric, ConversationCompletenessMetric
from deepeval.test_case import ConversationalTestCase, Turn


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

turnRelevancyMetric = TurnRelevancyMetric(threshold=0.5)
retentionMetric = KnowledgeRetentionMetric(threshold=0.5)
completnessMetric = ConversationCompletenessMetric(threshold=0.5)

test_case = ConversationalTestCase(
    turns = turns
)

evaluate(test_cases = [test_case], metrics = [turnRelevancyMetric,retentionMetric,completnessMetric])