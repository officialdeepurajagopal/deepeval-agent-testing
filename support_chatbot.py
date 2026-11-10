"""
support_chatbot.py
==================
A multi-turn airline customer-support chatbot for AeroNova Airways, with tool
calling, built with Claude (Anthropic Messages API).

Tools available (same data as booking_agent_plain.py):
  - get_booking_status(booking_ref)   → flight, route, date, seat and status
  - get_baggage_allowance(fare_class) → checked + cabin baggage per fare class

Unlike booking_agent_plain.py (which delegates the tool-call loop to LangGraph),
this chatbot manages the loop manually so the full message history — including
tool calls and tool results — is preserved across turns. This is what makes
multi-turn evaluation possible with DeepEval.

chat_turn() returns:
  - reply        : the final assistant text shown to the user
  - history      : updated message history to pass into the next turn
  - tools_called : list of (tool_name, args, result) for this turn

Usage:
    history = []
    reply, history, tools = chat_turn("What's the status of booking ANX7Q2?", history)
    reply, history, tools = chat_turn("How much baggage can I take?", history)

How to run standalone:
    python support_chatbot.py
"""

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

client = Anthropic()          # reads ANTHROPIC_API_KEY from the environment
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# In-memory data (same as booking_agent_plain.py)
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

# ---------------------------------------------------------------------------
# Tool definitions — passed to the Anthropic Messages API
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_booking_status",
        "description": "Look up a flight booking by its 6-character booking reference (PNR).",
        "input_schema": {
            "type": "object",
            "properties": {
                "booking_ref": {
                    "type": "string",
                    "description": "The booking reference, e.g. ANX7Q2",
                }
            },
            "required": ["booking_ref"],
        },
    },
    {
        "name": "get_baggage_allowance",
        "description": "Return the checked and cabin baggage allowance for a fare class.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fare_class": {
                    "type": "string",
                    "description": "Fare class: economy, premium economy or business",
                }
            },
            "required": ["fare_class"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------
def _execute_tool(name: str, args: dict) -> str:
    if name == "get_booking_status":
        ref = args.get("booking_ref", "").strip().upper()
        b = BOOKINGS.get(ref)
        if not b:
            return f"No booking found with reference {ref}."
        return (
            f"Booking {ref} for {b['passenger']}: flight {b['flight']}, {b['route']}, "
            f"on {b['date']} departing {b['departure']}. Status: {b['status']}. "
            f"Seat: {b['seat']}. Fare class: {b['fare_class'].replace('_', ' ')}."
        )

    if name == "get_baggage_allowance":
        key = args.get("fare_class", "").strip().lower().replace(" ", "_").replace("-", "_")
        allowance = BAGGAGE_ALLOWANCES.get(key)
        if not allowance:
            return f"No baggage allowance on file for fare class '{args.get('fare_class')}'."
        return allowance

    return f"Unknown tool: {name}"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a friendly and professional customer-support chatbot \
for AeroNova Airways, an international airline.

You have access to two tools:
  - get_booking_status   : use whenever a customer asks about a specific booking or flight
  - get_baggage_allowance: use whenever a customer asks about baggage limits

Rules:
- Always use the tools when you have the information needed to call them.
- Be concise and polite.
- Never discuss topics outside of AeroNova Airways customer support.
- Remember everything the customer tells you in the current conversation."""


# ---------------------------------------------------------------------------
# chat_turn() — one user turn, returns reply + updated history + tools used
# ---------------------------------------------------------------------------
def chat_turn(
    user_message: str,
    history: list[dict],
) -> tuple[str, list[dict], list[dict]]:
    """
    Send one user message, handle any tool calls, return the final reply.

    Returns:
        reply        : final assistant text
        history      : updated message list (include in next call)
        tools_called : list of {"name": ..., "args": ..., "result": ...}
                       for all tools the model called this turn
    """
    history = history + [{"role": "user", "content": user_message}]
    tools_called = []

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,          # system prompt is a top-level param, not a message
            messages=history,
            tools=TOOLS,
        )

        # Keep the assistant's full content (text + tool_use blocks) in history,
        # so the matching tool_result blocks in the next message are valid.
        history = history + [{"role": "assistant", "content": response.content}]

        # No tool call — final answer
        if response.stop_reason != "tool_use":
            reply = "".join(b.text for b in response.content if b.type == "text")
            return reply, history, tools_called

        # Execute every tool the model requested; all results go back
        # together in a single user message, then loop back.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = _execute_tool(block.name, block.input)
            tools_called.append({
                "name":   block.name,
                "args":   block.input,
                "result": result,
            })
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result,
            })
        history = history + [{"role": "user", "content": tool_results}]


# ---------------------------------------------------------------------------
# Standalone interactive demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("AeroNova Airways Support Chatbot (with tools) — type 'quit' to exit\n")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        reply, history, tools = chat_turn(user_input, history)
        if tools:
            for t in tools:
                print(f"  [tool] {t['name']}({t['args']}) → {t['result']}")
        print(f"Bot: {reply}\n")
