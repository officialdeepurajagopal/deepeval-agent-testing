"""
booking_agent.py
================
SAME agent as booking_agent_plain.py, with DeepEval observability added,
plus one extra tool (search_flights) loaded from the MCP server.

What changed vs booking_agent_plain.py? Only 4 things — all marked with "NEW":

  (1) Import CallbackHandler from deepeval.integrations.langchain
  (2) Build the callback handler once (no metrics baked in — metrics are
      attached by the test file at run time)
  (3) Pass it via config={"callbacks": [...]} on .invoke()
  (4) (Optional) Add @observe on the outer Python wrapper so any
      non-LangChain code we add later (guardrails, post-processing)
      also shows up as spans

That's it. Four lines, and now QA can run component-level evals
without ever touching this file again.
"""

import asyncio
import os

import nest_asyncio
nest_asyncio.apply()

import sys
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
from langchain_core.tools import tool
from langchain.agents import create_agent

# --- (1) NEW: import DeepEval tracing primitives --------------------------
from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace


# ---------------------------------------------------------------------------
# Same fake data as booking_agent_plain.py.
# ---------------------------------------------------------------------------
BOOKINGS = {
    "ANX7Q2": {
        "passenger": "Priya Menon", "flight": "AN 214",
        "route": "Bengaluru (BLR) → Singapore (SIN)",
        "date": "2026-12-18", "departure": "08:40",
        "status": "Confirmed", "seat": "14C", "fare_class": "economy",
    },
    "BKZ9M4": {
        "passenger": "James Carter", "flight": "AN 501",
        "route": "London (LHR) → New York (JFK)",
        "date": "2026-12-20", "departure": "11:15",
        "status": "Checked-in", "seat": "3A", "fare_class": "business",
    },
    "CLW3P8": {
        "passenger": "Aiko Tanaka", "flight": "AN 77",
        "route": "Dubai (DXB) → Sydney (SYD)",
        "date": "2026-12-22", "departure": "22:05",
        "status": "Cancelled", "seat": "-", "fare_class": "premium_economy",
    },
}

BAGGAGE_ALLOWANCES = {
    "economy":         "Economy: 1 checked bag up to 23 kg, plus 1 cabin bag up to 7 kg.",
    "premium_economy": "Premium Economy: 2 checked bags up to 23 kg each, plus 1 cabin bag up to 10 kg.",
    "business":        "Business: 2 checked bags up to 32 kg each, plus 2 cabin bags up to 10 kg combined.",
}


def _normalise_fare_class(fare_class: str) -> str:
    return fare_class.strip().lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Same tools as booking_agent_plain.py.
# ---------------------------------------------------------------------------
@tool
def get_booking_status(booking_ref: str) -> str:
    """Look up a flight booking by its 6-character booking reference (PNR)."""
    b = BOOKINGS.get(booking_ref.strip().upper())
    if not b:
        return f"No booking found with reference {booking_ref}."
    return (
        f"Booking {booking_ref.upper()} for {b['passenger']}: flight {b['flight']}, "
        f"{b['route']}, on {b['date']} departing {b['departure']}. "
        f"Status: {b['status']}. Seat: {b['seat']}. "
        f"Fare class: {b['fare_class'].replace('_', ' ')}."
    )


@tool
def get_baggage_allowance(fare_class: str) -> str:
    """Return the baggage allowance for a fare class (economy, premium economy, business)."""
    allowance = BAGGAGE_ALLOWANCES.get(_normalise_fare_class(fare_class))
    if not allowance:
        return f"No baggage allowance on file for fare class '{fare_class}'."
    return allowance


# ---------------------------------------------------------------------------
# Same agent as booking_agent_plain.py.
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-6")

# Load one extra tool (search_flights) from the MCP server at startup.
_mcp_tools = asyncio.run(
    MultiServerMCPClient({
        "aeronova": {
            "command": sys.executable,
            "args": [os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")],
            "transport": "stdio",
        }
    }).get_tools()
)

SYSTEM_PROMPT = (
    "You are a friendly booking-support agent for AeroNova Airways. "
    "Use the available tools to answer booking status, baggage and flight search questions. "
    "Keep replies short and helpful."
)

agent = create_agent(
    model=llm,
    tools=[get_booking_status, get_baggage_allowance] + _mcp_tools,
    system_prompt=SYSTEM_PROMPT,
)


# --- (2) NEW: a single shared callback handler ----------------------------
# This handler captures every LangChain LLM call, tool call, and chain step
# as a DeepEval span automatically. No per-function decoration needed for
# anything LangChain owns.
deepeval_callback = CallbackHandler()


def run_booking_agent(user_input: str) -> str:
    """Run the agent on a user message and return the final reply."""
    # --- (3) NEW: pass the callback handler in the .invoke() config ------
    result = asyncio.run(agent.ainvoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": [deepeval_callback]},
    ))
    reply = result["messages"][-1].content
    update_current_trace(output=reply)
    return reply


if __name__ == "__main__":
    print(run_booking_agent("What's the status of my booking ANX7Q2?"))
