# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Domain

All agents serve **AeroNova Airways**, a fictional airline. Shared fake data (keep in sync across
`booking_agent_plain.py`, `booking_agent.py` and `support_chatbot.py`):

- `BOOKINGS`: `ANX7Q2` (AN 214 BLR→SIN, Confirmed, economy), `BKZ9M4` (AN 501 LHR→JFK, Checked-in,
  business), `CLW3P8` (AN 77 DXB→SYD, Cancelled, premium economy)
- `BAGGAGE_ALLOWANCES` keyed by `economy` / `premium_economy` / `business`
- `mcp_server.py` `SCHEDULE` for `search_flights(origin, destination)`
- `policy_rag_agent.py` `POLICY_DOCS` and `policies.txt` hold the airline policies (cancellations, changes,
  baggage, check-in, delays, pets, special assistance). Test `expected_output`s must match these texts.

## Setup

```bash
pip install -r requirements.txt
echo "$PWD" > .venv/lib/python3.13/site-packages/deepeval_agent_demo.pth   # makes root modules importable from subfolders
```

`.env` (gitignored):
```
ANTHROPIC_API_KEY=sk-ant-...        # agents + judge (Claude Sonnet 4.6)
CONFIDENT_API_KEY=confident_us_...  # optional: stream traces to Confident AI
```

## Running

```bash
python booking_agent.py | policy_rag_agent.py | support_chatbot.py   # sanity-check an agent

python tests/test_tracingComponents.py        # TaskCompletion + ToolCorrectness
python tests/test_multipleEvalsTest.py        # PromptAlignment + StepEfficiency
python tests/test_customMetricEvals.py        # GEval correctness
python ragAgentTests/test_rag_agent.py        # ContextualPrecision + ContextualRecall + Faithfulness
python testsChatbots/test_Chatbot.py          # TurnRelevancy + KnowledgeRetention + ConversationCompleteness
python testsChatbots/test_ChatbotCustEvals.py # ConversationalGEval with TOOLS_CALLED
python test_testDataGeneration.py             # Synthesizer goldens + Bias/PIILeakage/Toxicity
```

Files are scripts (evaluation runs under `if __name__ == "__main__"` or at module level), not pytest tests.
`evals/` holds older course exercises; several still use `model="gpt-4o"` and need OpenAI credits.

## Agents

| File | Notes |
|---|---|
| `booking_agent_plain.py` | LangChain `create_agent` + Claude, tools `get_booking_status`, `get_baggage_allowance`. No eval code. |
| `booking_agent.py` | Same + 4 lines of DeepEval instrumentation + MCP `search_flights` tool. Entry point `run_booking_agent()`. |
| `policy_rag_agent.py` | Tool `search_policies` → `retrieve()`: Claude Sonnet ranks the 9 docs and returns the top 3 (no embeddings — Anthropic has no embeddings API). Entry point `run_policy_rag_agent()` sets `retrieval_context` on the trace. |
| `support_chatbot.py` | Anthropic Messages API with a manual tool-use loop. `chat_turn(msg, history)` → `(reply, history, tools_called)`; history holds Anthropic content blocks. |

## Patterns

**Booking/RAG agent tests**: wrap the agent in `@observe(name="booking_agent")` (or `"policy_rag_agent"`),
copy `expected_tools` / `expected_output` from `get_current_golden()` onto the trace with
`update_current_trace(...)`, and run via `dataset.evals_iterator(metrics=...)`. DeepEval reads these
fields from the trace, not from the golden.

**Chatbot tests**: build `ConversationalTestCase(turns=[Turn(...)])` by driving `chat_turn()` live, pass
`tools_called=[ToolCall(...)]` on assistant turns, and call `evaluate()`. No trace needed.

## Gotchas

- Pass judges as `AnthropicModel(model="claude-sonnet-4-6")`. A plain model string makes DeepEval use OpenAI.
- The installed Anthropic SDK / langchain-anthropic reject `temperature` for Claude Sonnet 4.6 — don't pass it.
- `GEval`, `TaskCompletionMetric`, `ToolCorrectnessMetric` are single-turn only; use `ConversationalGEval`
  and conversational metrics with `ConversationalTestCase`.
- `ToolCorrectnessMetric` requires `expected_tools`, so it can't score Synthesizer goldens.
- `Synthesizer.generate_goldens_from_docs()` needs an embedder (OpenAI by default); use
  `generate_goldens_from_contexts()` with `policies.txt` split by section instead.
- Never import from `evals/` modules — importing one runs its whole evaluation.

## Metric reference

| Metric | Needs `expected_output`? | Needs `retrieval_context`? | LLM judge? |
|---|---|---|---|
| `TaskCompletionMetric` | No | No | Yes |
| `ToolCorrectnessMetric` | No (needs `expected_tools`) | No | No (name match) |
| `AnswerRelevancyMetric` | No | No | Yes |
| `PromptAlignmentMetric` | No | No | Yes |
| `StepEfficiencyMetric` | No | No | Yes |
| `GEval` (Correctness) | **Yes** | No | Yes |
| `FaithfulnessMetric` | No | **Yes** | Yes |
| `ContextualPrecisionMetric` | **Yes** | **Yes** | Yes |
| `ContextualRecallMetric` | **Yes** | **Yes** | Yes |
| `KnowledgeRetentionMetric` | No | No | Yes |
| `RoleAdherenceMetric` | No | No | Yes |
| `TurnRelevancyMetric` | No | No | Yes |
| `ConversationCompletenessMetric` | No | No | Yes |
| `ConversationalGEval` | No | No | Yes |
| `BiasMetric` | No | No | Yes |
| `ToxicityMetric` | No | No | Yes |
| `PIILeakageMetric` | No | No | Yes |

## Known issues / patches

- **DeepEval 4.0.4 `_make_hashable` bug** (historical): `ToolMessage` objects in `tools_called` were unhashable and
  crashed `ToolCorrectnessMetric`; it was patched in the venv's `deepeval/test_case/llm_test_case.py`. The current
  venv runs DeepEval 4.3.3, where `ToolCorrectnessMetric` works without the patch.
- **`create_react_agent` deprecation**: use `from langchain.agents import create_agent` with `system_prompt=`.
