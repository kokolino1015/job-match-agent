# Job posting agent

A tool-calling agent that fetches job postings, decides which are worth a closer look, and scores the promising ones against my CV. The loop runs locally on Ollama; the tools live in a separate MCP server, and scoring calls out to a separate service.

## How it works

```
fetch.py         Lever API -> normalised postings.json, deduplicated against seen.json
main.py          hand-written agent loop on Ollama: list -> read -> score -> ranked recommendation
graph_agent.py   the same agent rebuilt in LangGraph, with human approval before scoring;
                 gets its tools from jobs_server.py over MCP
jobs_server.py   MCP server exposing the three tools (list, read, score)
test_interrupts.py  tests the approval gate with a fake tool: approved calls run once,
                 declined calls never run
```

The agent has three tools (in `jobs_server.py` for the LangGraph version):

- `list_posting` returns id, title, company and location for every saved posting. Cheap, so the model sees the whole landscape without spending context on descriptions.
- `read_posting` returns one full description. The model calls this only for postings worth a closer look.
- `score_posting` sends the CV and the posting to [cv-analyzer](https://github.com/kokolino1015/cv-analyzer) and returns a structured match score.

The split is deliberate: the model decides which postings deserve the expensive operations, rather than the code scoring everything.

A system prompt holds the search criteria (stack preferences, what to rule out, seniority bounds), so the filtering is configuration rather than code.

## Running it

```bash
pip install -r requirements.txt
ollama pull qwen3:30b
python fetch.py     # populate postings.json
python main.py      # run the hand-written loop
python graph_agent.py   # or the LangGraph version, which asks before scoring
python test_interrupts.py   # approval gate test, no model or server needed
```

`graph_agent.py` starts `jobs_server.py` itself, so there is nothing else to launch. To look at the server on its own, open it in the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector .venv/bin/python jobs_server.py
```

Needs `.env` with `ANTHROPIC_API_KEY` (used by cv-analyzer, not by the loop), `cv.txt`, `profile.txt`, and `companies.json`. Example files are included for the last two. `score_posting` needs cv-analyzer running on localhost:8000.

## Notes on the two tool protocols

The loop was first written against the Anthropic Messages API, then ported to Ollama. `TOOLS` and `run_tool` were unchanged; only the schema format and the loop differ.

| | Anthropic | Ollama |
|---|---|---|
| Schema key | `input_schema` | `parameters`, wrapped in `{"type": "function", ...}` |
| Continue signal | `stop_reason == "tool_use"` | non-empty `message.tool_calls` |
| Truncation signal | `stop_reason == "max_tokens"` | `done_reason == "length"` |
| Matching results to calls | explicit `tool_use_id` | positional order |
| Error flag | `is_error` on the result block | none, folded into the text |

Ollama has no tool call ids, so results are matched by order. Tool results include their arguments in the text so the model can tell them apart regardless.

Tool support is per-model, not per-runtime: gemma3 has no `tools` capability, qwen3 does.

## The LangGraph version

`graph_agent.py` rebuilds the loop as a graph, to see what a framework takes over from hand-written code.

```
START -> agent -> should_continue -> approval -> gate -> agent -> ... -> END
                        |
                        +-> END   (no tool calls left)
```

- `agent` calls the model with the tools bound.
- `approval` looks at the pending tool calls and, if any of them need approval, asks about all of them in one `interrupt()`. It only asks; it never runs anything.
- `gate` runs the tools. It never pauses.

| Hand-written loop (`main.py`) | LangGraph (`graph_agent.py`) |
|---|---|
| `messages` list appended by hand | `MessagesState`, merged by the `add_messages` reducer |
| JSON schemas written by hand | `@tool` builds them from the name, docstring and `Annotated` hints |
| `run_tool` dispatcher and error prefix | `ToolNode(tools, handle_tool_errors=True)`, called from the `gate` node |
| `while` loop and `if tool_calls` | conditional edge `should_continue` returning `"approval"` or `END` |
| `MAX_TURNS` guard | `recursion_limit` in the run config |
| no pause possible | `interrupt()` plus a checkpointer, in its own `approval` node |

### Human in the loop

`score_posting` calls a paid API, so a human approves it first. Which tools need approval is configured in one place, `need_approval = {"score_posting"}`; the tools themselves contain no approval logic.

1. **`approval` asks.** It collects the pending calls whose name is in `need_approval` and calls `interrupt()` once, with a list of `{message, id}` items, one per risky call. The graph pauses and the checkpointer (`InMemorySaver`, keyed by `thread_id`) keeps the state. `invoke` returns a result with an `__interrupt__` key.
2. **The caller answers.** The main loop asks y/n for each item and resumes with `Command(resume={tool_call_id: "y" or "n"})`. That dict is what `interrupt()` returns inside the node, and the node stores it in the state as `decisions`.
3. **`gate` acts.** It splits the tool calls: a call is declined if it needs approval and its decision is not `"y"`. Declined calls get a tool message telling the model not to estimate a score and to report the posting as not scored. The approved calls go to `ToolNode` in a copy of the AI message that contains only them (`model_copy(update={"tool_calls": approved})`), so ToolNode keeps doing what it does well (dispatch by name, parallel execution, errors returned as messages) but can only see what was allowed.

Approval fails closed: an empty answer counts as no, and a missing decision counts as no.

Every tool call still gets exactly one result message, which the model APIs require.

### What testing it showed

**The first design ran an approved call twice.** Originally, `score_posting` called `interrupt()` itself and then made the POST. LangGraph resumes a paused node by rerunning it from the top, handing the saved answers back to `interrupt()`. With two score calls in one step, the first call's POST had already happened when the second call's `interrupt()` paused the step, so the step's results were discarded and the POST ran again on resume. With n approved calls in one step, that is n(n+1)/2 executions instead of n.

A test with the model faked out confirmed it: a small graph containing only `ToolNode`, fed a hand-built `AIMessage` with two tool calls. The approved POST printed in the second invoke and again in the third.

**The rule that came out of it:** a step that can pause must have no side effects, and a step with side effects must never pause. Hence the split into `approval` (pauses, does nothing) and `gate` (acts, never pauses).

**The fix was tested the same way**, with a fake model asking to read posting A and score A and B, and a fake POST. `test_interrupts.py` keeps the core of it as a committed test (approval + gate with a fake `score_posting`, one approved, one declined):

| Answers | POSTs | Tool results the model receives |
|---|---|---|
| y, y | A once, B once | the description, both scores |
| y, n | A once | the description, A's score, "declined" for B |
| n, n | none | the description, "declined" for both |

The read call is never blocked, because it is not in `need_approval`.

## The tools as an MCP server

The LangGraph agent no longer defines its tools. They live in `jobs_server.py`, an [MCP](https://modelcontextprotocol.io) server, and the agent connects to it as a client.

**Why:** the tools are exposed once through a standard protocol, so any MCP client can use them (this agent, Claude Desktop, an IDE), and each side can be tested on its own: the tools in the MCP Inspector, the approval logic with a fake tool. It also gives two separate boundaries. The server decides what is possible at all: helpers like `read_file` are plain functions, not tools, so no client can ever call them. The agent decides, per call, whether a human approved it.

**How it fits together:**

```
graph_agent.py ── MultiServerMCPClient ── stdio ── jobs_server.py (FastMCP)
  approval + gate stay here                          list / read / score live here
```

- **Transport: stdio.** The agent starts the server as a child process (`sys.executable jobs_server.py`) and they exchange JSON-RPC messages over stdin/stdout. No port, no network, no auth to build, and the server lives exactly as long as the agent's session. If the tools had to be shared or deployed separately, the switch would be to Streamable HTTP plus authentication; the graph and gate would not change.
- **One session per run.** `client.session("Main")` keeps one server process for the whole run and `load_mcp_tools(session)` turns its tools into ordinary LangChain tools, so `bind_tools`, `ToolNode` and `need_approval` work unchanged. (`client.get_tools()` starts a fresh server process for every tool call: the log shows a `ListToolsRequest` before each `CallToolRequest`.) Everything that uses the tools runs inside the `async with` block.
- **Async.** MCP tools are async-only, so the gate awaits `tool_node.ainvoke(...)`, the graph runs with `ainvoke`, and `main()` is started once with `asyncio.run`. The graph is built in `build_graph(tools)` after the tools are loaded.
- **Testable gate.** `make_gate(tool_node)` builds the gate around any `ToolNode`. The agent passes the MCP tools; `test_interrupts.py` passes a fake `score_posting` with the same name and checks that an approved call runs once and a declined one never runs.

**What went wrong on the way:**

- **Version conflict.** In MCP SDK 2.x `FastMCP` was renamed `MCPServer`, but `langchain-mcp-adapters` (0.3.2) requires `mcp<2`. The SDK is pinned to 1.x.
- **Lost parameter descriptions.** LangChain's `@tool` reads a bare string in `Annotated[str, "..."]` as the description; the MCP SDK builds schemas with Pydantic, which ignores it. The descriptions use `Field(description=...)`, and the Inspector confirmed they reach the schema. That sentence is what tells the model to use ids exactly as listed.
- **A server that never started.** With `pass` instead of `mcp.run()`, `mcp run jobs_server.py` still worked (the CLI starts the server itself) but `python jobs_server.py` exited at once, so the Inspector showed no connection and no error.

## Known limitations

- **Failed tools used to be invisible.** A 422 from the scoring service returned an error body with a 200-shaped `.json()`, so it reached the model looking like a result. The model noticed, said so in its reasoning, and then produced a confident ranking from scores it never received. Fixed with `raise_for_status()`; the lesson is that a tool failure the harness does not flag becomes a fabricated answer.
- **Local models need instructions spelled out.** Scoring was described as an option ("score the two or three strongest") and simply never happened. It had to become a requirement.
- **Tool descriptions are prompts.** A vague description means an unused tool.
- **One source.** Lever only. Most companies use other systems, and several have no readable API at all, so they are marked `manual` in the config and checked by hand.
- **`postings.json` holds one run's new postings**, so running the fetcher twice before scoring loses the batch.
- **No evaluation set.** Results were checked by comparing the agent's rankings against my own judgment on four postings I had already assessed. That caught a real bug (a junior-titled role dismissed on the title alone) but it is four data points, not a benchmark.
- **Approval lives in the terminal.** The resume loop uses `input()`. A real deployment would put the question where the person already is (a ticket, a chat message) and resume when they answer.
- **No idempotency key on scoring.** The approval split stops the replay, but the scoring call itself is still not safe to repeat. An idempotency key would make an accidental second call harmless.
- **`input()` inside async code** blocks the event loop while waiting for an answer. Harmless here, because nothing else runs meanwhile; in a server with many users it would freeze everything.
- **Scoring has to be asked for.** With qwen3 the prompt has to say "score at least the two best postings", or the model skips the paid tool. Same lesson as above.
- **The final answer is not checked against the tool results.** In one run the model called AJW's match "production AI deployment experience" while the scoring itself listed limited production deployment as a gap, and it gave reasons for postings it had not read. That belongs in an eval set.