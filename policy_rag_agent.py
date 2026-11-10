"""
policy_rag_agent.py
============
An AeroNova Airways customer-support agent that answers questions by retrieving from a small
in-memory airline policy knowledge base instead of calling fixed lookup functions.

Architecture:
  - Knowledge base: 9 policy documents held in memory
  - Retriever: Claude (Sonnet) reads the numbered docs and picks the top-3 most
    relevant to the query. Anthropic has no embeddings API, so the LLM itself
    does the ranking — practical for a small knowledge base like this one.
  - Tool: search_policies(query) — returns the top-3 chunks
  - LLM: Claude Sonnet answers using only the retrieved chunks
  - Instrumentation: DeepEval CallbackHandler + @observe, same 4-line pattern

What's new vs booking_agent.py:
  - search_policies tool replaces get_booking_status / get_baggage_allowance
  - Retrieved chunks are captured into the DeepEval trace as retrieval_context
    so that RAG metrics (Faithfulness, Contextual Precision/Recall) can run
"""
import json
import re

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace


# ---------------------------------------------------------------------------
# Knowledge base — AeroNova Airways policies as plain-text documents.
# ---------------------------------------------------------------------------
POLICY_DOCS = [
    # Cancellations and changes
    "Cancellation and refund policy: Flexi fares can be cancelled up to 24 hours before "
    "departure for a full refund. Saver fares are non-refundable, but taxes and airport fees "
    "are refunded on request. Refunds are returned to the original payment method within "
    "7–10 business days.",

    "Flight change policy: Flights can be changed up to 3 hours before departure through "
    "Manage Booking. Flexi fares change free of charge; Saver fares pay a $75 change fee plus "
    "any fare difference. Name changes are not permitted.",

    # Baggage
    "Checked baggage policy: Economy includes 1 checked bag up to 23 kg. Premium Economy "
    "includes 2 checked bags up to 23 kg each. Business includes 2 checked bags up to 32 kg "
    "each. No single bag may exceed 158 cm in total dimensions.",

    "Excess and overweight baggage policy: Extra bags cost $60 each when added online at "
    "least 24 hours before departure, or $90 at the airport. Bags weighing 23–32 kg incur a "
    "$50 overweight fee. Bags over 32 kg are not accepted as checked baggage.",

    "Cabin baggage policy: Economy passengers may bring 1 cabin bag up to 7 kg plus 1 small "
    "personal item such as a handbag or laptop bag. Premium Economy and Business passengers "
    "may bring cabin bags up to 10 kg. Cabin bags must fit 56 x 36 x 23 cm.",

    # Airport
    "Check-in policy: Online check-in opens 48 hours and closes 90 minutes before departure. "
    "Airport check-in counters close 60 minutes before domestic flights and 75 minutes before "
    "international flights. Boarding gates close 20 minutes before departure.",

    "Delays and compensation policy: For delays over 3 hours, AeroNova provides meal vouchers. "
    "For delays over 5 hours, passengers may cancel for a full refund. If a delay requires an "
    "overnight stay, AeroNova provides hotel accommodation and transfers at no cost.",

    # Special travel
    "Travelling with pets policy: Small cats and dogs up to 8 kg including the carrier may "
    "travel in the cabin for $120 per flight. Larger pets travel in the hold. Pets must be "
    "booked at least 48 hours before departure; only 2 cabin pets are allowed per flight.",

    "Special assistance policy: Wheelchair assistance and support for passengers with reduced "
    "mobility is free and should be requested at least 48 hours before departure. Children "
    "aged 5–11 travelling alone must use the unaccompanied minor service, which costs $100 "
    "per flight.",
]


# ---------------------------------------------------------------------------
# Retriever — Claude ranks the documents (Anthropic has no embeddings API).
# ---------------------------------------------------------------------------
_retriever_client = Anthropic()          # reads ANTHROPIC_API_KEY
RETRIEVER_MODEL = "claude-sonnet-4-6"


def retrieve(query: str, k: int = 3) -> list[str]:
    """Return the k policy documents most relevant to the query, best first."""
    numbered = "\n".join(f"[{i}] {doc}" for i, doc in enumerate(POLICY_DOCS))
    response = _retriever_client.messages.create(
        model=RETRIEVER_MODEL,
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": (
                f"Documents:\n{numbered}\n\n"
                f"Query: {query}\n\n"
                f"Return the indices of the {k} documents most relevant to the query, "
                f"most relevant first, as a JSON array of integers (e.g. [4, 0, 7]). "
                f"Reply with the JSON array only."
            ),
        }],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    match = re.search(r"\[[\d,\s]*\]", text)
    indices = json.loads(match.group()) if match else []
    valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(POLICY_DOCS)]
    return [POLICY_DOCS[i] for i in dict.fromkeys(valid)][:k]


# ---------------------------------------------------------------------------
# Retrieval tool — returns top-3 policy chunks for a query.
# Also stashes the retrieved text in a module-level list so the @observe
# wrapper can forward it to the DeepEval trace as retrieval_context.
# ---------------------------------------------------------------------------
_last_retrieved: list[str] = []


@tool
def search_policies(query: str) -> str:
    """Search the AeroNova Airways policy knowledge base (cancellations, baggage, check-in, delays, pets, assistance)."""
    global _last_retrieved
    chunks = retrieve(query, k=3)
    _last_retrieved = chunks
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-6")

agent = create_agent(
    model=llm,
    tools=[search_policies],
    system_prompt=(
        "You are a friendly customer-support agent for AeroNova Airways. "
        "Use the search_policies tool to look up cancellation, baggage, check-in and travel policies. "
        "Answer only from the retrieved information. Keep replies concise."
    ),
)

# DeepEval callback handler — captures LangChain spans automatically.
deepeval_callback = CallbackHandler()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_policy_rag_agent(user_input: str) -> str:
    """Run the RAG agent and return the final reply."""
    global _last_retrieved
    _last_retrieved = []

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": [deepeval_callback]},
    )
    reply = result["messages"][-1].content

    # Set clean reply string and retrieved chunks on the trace.
    update_current_trace(
        output=reply,
        retrieval_context=_last_retrieved if _last_retrieved else None,
    )

    return reply


if __name__ == "__main__":
    print(run_policy_rag_agent("Can I get a refund if I cancel my flight?"))
