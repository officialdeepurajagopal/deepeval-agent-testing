# DeepEval Agent Testing — AeroNova Airways

Evaluating three kinds of AI agents for a fictional airline, **AeroNova Airways**,
with [DeepEval](https://deepeval.com). The agents run on Claude, and Claude is also
used as the LLM judge for every LLM-based metric.

| Agent | What it does | How it's evaluated |
|---|---|---|
| **Booking agent** (`booking_agent.py`) | Tool-calling agent: looks up bookings by reference, returns baggage allowances, and searches flights via an MCP server | Task Completion, Tool Correctness, Prompt Alignment, Step Efficiency, GEval correctness |
| **Policy RAG agent** (`policy_rag_agent.py`) | Answers policy questions (cancellations, baggage, check-in, delays, pets, assistance) from a 9-document knowledge base | Contextual Precision, Contextual Recall, Faithfulness |
| **Support chatbot** (`support_chatbot.py`) | Multi-turn chatbot with its own tool-call loop, remembering context across turns | Turn Relevancy, Knowledge Retention, Conversation Completeness, Conversational GEval |

Plus synthetic test generation from `policies.txt` and safety evals (Bias, Toxicity, PII Leakage).

---

## Sample data

| Booking ref | Passenger | Flight | Route | Date | Status | Fare class |
|---|---|---|---|---|---|---|
| `ANX7Q2` | Priya Menon | AN 214 | Bengaluru (BLR) → Singapore (SIN) | 2026-12-18 08:40 | Confirmed | Economy |
| `BKZ9M4` | James Carter | AN 501 | London (LHR) → New York (JFK) | 2026-12-20 11:15 | Checked-in | Business |
| `CLW3P8` | Aiko Tanaka | AN 77 | Dubai (DXB) → Sydney (SYD) | 2026-12-22 22:05 | Cancelled | Premium Economy |

All data is in-memory and fictional, so the agents' tool results are deterministic.

---

## Project structure

| Path | Purpose |
|---|---|
| `booking_agent_plain.py` | The booking agent with **no** eval code — the "before" state |
| `booking_agent.py` | The same agent plus 4 lines of DeepEval instrumentation and the MCP flight-search tool. Tests import from here. |
| `mcp_server.py` | MCP server exposing `search_flights(origin, destination)` |
| `policy_rag_agent.py` | RAG agent. Claude ranks the policy documents to retrieve the top 3 (Anthropic has no embeddings API), then answers from them |
| `support_chatbot.py` | Multi-turn chatbot on the Anthropic Messages API; `chat_turn()` returns `(reply, history, tools_called)` |
| `policies.txt` | Airline policy document used to synthesize test goldens |
| `tests/` | Booking agent evals |
| `ragAgentTests/` | RAG agent evals |
| `testsChatbots/` | Chatbot evals |
| `test_testDataGeneration.py` | Generates goldens from `policies.txt` with DeepEval's Synthesizer, then runs safety metrics |
| `evals/` | Earlier course exercises (some still use GPT-4o judges and need an OpenAI key) |

---

## The instrumentation diff

The difference between `booking_agent_plain.py` and `booking_agent.py` is 4 lines:

1. `from deepeval.integrations.langchain import CallbackHandler`
2. `deepeval_callback = CallbackHandler()`
3. `config={"callbacks": [deepeval_callback]}` on `.invoke()`
4. `@observe(name="booking_agent")` on the wrapper function (in the test file)

That is all the agent needs for DeepEval to capture every LLM call and tool call as a trace.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your ANTHROPIC_API_KEY
```

To run tests from any folder without `sys.path` tweaks, register the project root with the venv once:

```bash
echo "$PWD" > .venv/lib/python3.13/site-packages/deepeval_agent_demo.pth
```

---

## Running

```bash
# Sanity-check each agent on its own
python booking_agent.py
python policy_rag_agent.py
python support_chatbot.py              # interactive; type 'quit' to exit

# Booking agent evals
python tests/test_tracingComponents.py # Task Completion + Tool Correctness
python tests/test_multipleEvalsTest.py # Prompt Alignment + Step Efficiency
python tests/test_customMetricEvals.py # GEval correctness vs expected_output
python tests/test_TaskCompletion.py

# RAG agent evals
python ragAgentTests/test_rag_agent.py # Contextual Precision + Recall + Faithfulness

# Chatbot evals
python testsChatbots/test_Chatbot.py           # Turn Relevancy + Knowledge Retention + Completeness
python testsChatbots/test_ChatbotCustEvals.py  # Conversational GEval (incl. tool use)

# Synthetic goldens + safety metrics
python test_testDataGeneration.py
```

---

## Notes

- **Judge model:** every metric is given `AnthropicModel(model="claude-sonnet-4-6")`. A plain string like
  `model="claude-sonnet-4-6"` would make DeepEval use its OpenAI wrapper.
- **Single-turn vs multi-turn:** `GEval`, `TaskCompletionMetric` and `ToolCorrectnessMetric` work on
  `LLMTestCase`; conversations need `ConversationalGEval` and the conversational metrics.
- **Synthetic goldens:** `generate_goldens_from_docs()` needs an embedder (OpenAI by default), so
  `test_testDataGeneration.py` splits `policies.txt` into sections and uses `generate_goldens_from_contexts()`.
