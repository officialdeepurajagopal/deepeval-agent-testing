"""
mcp_server.py
=============
MCP server exposing one tool used by booking_agent.py: a flight search over
AeroNova Airways' (fake) schedule.

Run standalone to verify:
    python mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AeroNova Flight Search")

# (origin, destination) IATA codes -> scheduled flights
SCHEDULE = {
    ("BLR", "SIN"): [
        "AN 214 departs 08:40, arrives 15:55 — Economy from $289, Business from $1,140",
        "AN 218 departs 23:10, arrives 06:20+1 — Economy from $245, Business from $990",
    ],
    ("LHR", "JFK"): [
        "AN 501 departs 11:15, arrives 14:20 — Economy from $520, Business from $3,200",
        "AN 505 departs 18:45, arrives 21:50 — Economy from $480, Business from $2,950",
    ],
    ("DXB", "SYD"): [
        "AN 77 departs 22:05, arrives 18:30+1 — Economy from $760, Premium Economy from $1,450",
    ],
}


@mcp.tool()
def search_flights(origin: str, destination: str) -> str:
    """Search scheduled AeroNova flights between two airports (3-letter IATA codes, e.g. BLR, SIN)."""
    flights = SCHEDULE.get((origin.strip().upper(), destination.strip().upper()))
    if not flights:
        return f"No AeroNova flights found from {origin.upper()} to {destination.upper()}."
    return f"Flights from {origin.upper()} to {destination.upper()}:\n- " + "\n- ".join(flights)


if __name__ == "__main__":
    mcp.run()
