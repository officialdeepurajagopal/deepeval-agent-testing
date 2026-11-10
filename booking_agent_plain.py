"""
booking_agent_plain.py
======================
A simple airline booking-support agent for AeroNova Airways, built with
LangChain + Claude. No evaluation code anywhere. This is what a developer
would write if they'd never heard of DeepEval.

The agent has two tools:
  - get_booking_status(booking_ref)   -> flight, route, date, seat, status
  - get_baggage_allowance(fare_class) -> checked + cabin baggage rules

It uses Claude to decide which tool to call and to compose the answer.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()
from langchain_core.tools import tool
from langchain.agents import create_agent


# ---------------------------------------------------------------------------
# Fake in-memory data — keeps the demo offline and deterministic.
# Booking references (PNRs) are 6-character codes.
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
# Tools — plain Python functions exposed to the agent via @tool.
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
# The agent — a LangGraph ReAct agent powered by Claude.
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-6")

SYSTEM_PROMPT = (
    "You are a friendly booking-support agent for AeroNova Airways. "
    "Use the available tools to answer booking status and baggage questions. "
    "Keep replies short and helpful."
)

agent = create_agent(
    model=llm,
    tools=[get_booking_status, get_baggage_allowance],
    system_prompt=SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Single entry point — this is what test code will import.
# ---------------------------------------------------------------------------
def run_booking_agent(user_input: str) -> str:
    """Run the agent on a user message and return the final reply."""
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(run_booking_agent("What's the status of my booking ANX7Q2?"))
