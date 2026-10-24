# Job posting agent

A tool-calling agent that fetches job postings, decides which are worth a closer look, scores the promising ones against my CV, and gives a verdict (apply / maybe / skip) for every posting. The loop runs locally on Ollama; the tools live in a separate MCP server, and scoring calls out to a separate service. An eval set measures the verdicts against my own.

## How it works

```
agent/graph_agent.py     the agent in LangGraph, with human approval before scoring and a final
                         decide step; gets its tools from server/jobs_server.py over MCP
agent/models.py          the data shapes: State, Verdict (what the model returns),
                         Decision and AgentResult (what the code stores)
server/jobs_server.py    MCP server exposing the three tools (list, read, score)
fetch/fetch.py           Lever API -> normalised data/postings.json, deduplicated against seen.json
legacy/main.py           the first, hand-written agent loop on Ollama: list -> read -> score -> ranking
tests/test_interrupts.py tests the approval gate with a fake tool: approved calls run once,
                         declined calls never run
evals/run.py             runs the agent on the eval postings and compares its verdicts with mine
evals/*.example.json     the format of the eval cases and postings (the real ones are private)
data/                    postings.json, profile.txt, cv.txt, companies.json
config.py                shared settings (model, token limit, turn limit)
```

Paths are built from the location of each file (`Path(__file__)`), not from the working directory, so a script works no matter where it is started from. Running as a module from the root also puts the root on the import path. The server is the exception: the agent starts it as a plain script, so it adds the repository root to `sys.path` itself before importing from the root.

The agent has three tools (in `jobs_server.py` for the LangGraph version):

- `list_posting` returns id, title, company and location for every saved posting. Cheap, so the model sees the whole landscape without spending context on descriptions.
- `read_posting` returns one full description. The model calls this only for postings worth a closer look.
- `score_posting` sends the CV and the posting to [cv-analyzer](https://github.com/kokolino1015/cv-analyzer) and returns a structured match score.

The split is deliberate: the model decides which postings deserve the expensive operations, rather than the code scoring everything.

A system prompt holds the search criteria (stack preferences, what to rule out, seniority bounds), so the filtering is configuration rather than code.

## Running it

Run everything from the repository root, as modules:

```bash
pip install -r requirements.txt
ollama pull qwen3:30b
python -m fetch.fetch              # populate data/postings.json
python -m agent.graph_agent        # the agent, which asks before scoring
python -m legacy.main              # the original hand-written loop
python -m tests.test_interrupts    # approval gate test, no model or server needed
python -m evals.run                # eval: needs Ollama, not cv-analyzer
```

The agent starts `server/jobs_server.py` itself, so there is nothing else to launch. The server reads `data/postings.json` unless the `POSTINGS_FILE` environment variable points elsewhere. To look at the server on its own, open it in the MCP Inspector (the Inspector starts the server with its own environment, so variables go through `-e`):

```bash
npx @modelcontextprotocol/inspector .venv/bin/python server/jobs_server.py
npx @modelcontextprotocol/inspector -e POSTINGS_FILE=evals/postings_manual.json .venv/bin/python server/jobs_server.py
```

Needs `.env` with `ANTHROPIC_API_KEY` (used by cv-analyzer, not by the loop), and `data/cv.txt`, `data/profile.txt` and `data/companies.json`. `score_posting` needs cv-analyzer running on localhost:8000.

The eval cases and their postings are private (they contain notes on my own applications). To run the eval with the examples, copy `evals/cases.example.json` to `evals/cases.json` and `evals/postings_manual.example.json` to `evals/postings_manual.json`.

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

The LangGraph version does have ids with Ollama: `langchain-ollama` generates an id for each tool call on the client side, so `ToolMessage(tool_call_id=...)`, the approval answers keyed by call id, and the decide step's score matching all work the same way as with Anthropic. The ids exist in LangChain's messages, not in the Ollama API, so they help the code pair calls with results, not the model.

Tool support is per-model, not per-runtime: gemma3 has no `tools` capability, qwen3 does.

## The LangGraph version

`graph_agent.py` rebuilds the loop as a graph, to see what a framework takes over from hand-written code.

```
START -> agent -> should_continue -> approval -> gate -> agent -> ...
                        |
                        +-> decide -> END   (no tool calls left)
```

- `agent` calls the model with the tools bound.
- `approval` looks at the pending tool calls and, if any of them need approval, asks about all of them in one `interrupt()`. It only asks; it never runs anything.
- `gate` runs the tools. It never pauses.
- `decide` gives a verdict for every posting (see below).

| Hand-written loop (`main.py`) | LangGraph (`graph_agent.py`) |
|---|---|
| `messages` list appended by hand | `MessagesState`, merged by the `add_messages` reducer |
| JSON schemas written by hand | `@tool` builds them from the name, docstring and `Annotated` hints |
| `run_tool` dispatcher and error prefix | `ToolNode(tools, handle_tool_errors=True)`, called from the `gate` node |
| `while` loop and `if tool_calls` | conditional edge `should_continue` returning `"approval"` or `"decide"` |
| `MAX_TURNS` guard | `recursion_limit` in the run config |
| no pause possible | `interrupt()` plus a checkpointer, in its own `approval` node |
| free-text final answer | `decide` node with structured output, one verdict per posting |

### Human in the loop

`score_posting` calls a paid API, so a human approves it first. Which tools need approval is configured in one place, `need_approval = {"score_posting"}`; the tools themselves contain no approval logic.

1. **`approval` asks.** It collects the pending calls whose name is in `need_approval` and calls `interrupt()` once, with a list of `{message, id}` items, one per risky call. The graph pauses and the checkpointer (`InMemorySaver`, keyed by `thread_id`) keeps the state. `invoke` returns a result with an `__interrupt__` key.
2. **The caller answers.** The main loop asks y/n for each item and resumes with `Command(resume={tool_call_id: "y" or "n"})`. That dict is what `interrupt()` returns inside the node, and the node stores it in the state as `approvals`.
3. **`gate` acts.** It splits the tool calls: a call is declined if it needs approval and its answer is not `"y"`. Declined calls get a tool message telling the model not to estimate a score and to report the posting as not scored. The approved calls go to `ToolNode` in a copy of the AI message that contains only them (`model_copy(update={"tool_calls": approved})`), so ToolNode keeps doing what it does well (dispatch by name, parallel execution, errors returned as messages) but can only see what was allowed.

Approval fails closed: an empty answer counts as no, and a missing answer counts as no.

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

### The decide step

The agent loop decides what to read and what to score. The final verdicts come from a separate `decide` node, because the first version showed that leaving the reading to the model does not work.

**What went wrong first.** `decide` asked the model, once and with the whole history, for a verdict on every posting. The agent had made about three tool calls, so most postings were in the history only as a title. The model still had to judge them, so it guessed: "likely PHP/JS" for a backend role with Python, ".NET" for a Java role, a junior AI role dismissed as "likely low-code". On 7 postings: 4 of 7 agreed with mine, with 2 false skips.

**What it does now.** Reading every posting is a requirement, so code guarantees it instead of the model:

- `decide` calls `list_posting` itself and loops over the result. The model never chooses which postings to judge.
- For each posting it calls `read_posting` and makes one structured-output call with fresh messages: the profile, that posting's title, company, location and full text, and its score if the agent scored it ("not scored" otherwise). One posting per call also keeps every call well inside the context window, and one posting's text cannot leak into another's verdict.
- The model returns only `Verdict(verdict, reason)`. The posting id is attached by code, so the model cannot invent or miss an id. (It did both when it wrote the ids itself: one run returned `posting_id='siteground'`.)
- Scores are found in the history by matching each `score_posting` call's id to its tool result, since the result message only carries the call id. Declined calls and tool errors are skipped.

`decide` calls the MCP tools from code with `ainvoke`. These are the same tool objects `ToolNode` uses; the only difference is who chooses the call. Only the read-only tools are called this way: `score_posting` stays behind the approval gate.

Same 7 postings after the change: 5 of 7, 1 false skip, and every reason quotes the posting text.

## The tools as an MCP server

The LangGraph agent no longer defines its tools. They live in `jobs_server.py`, an [MCP](https://modelcontextprotocol.io) server, and the agent connects to it as a client.

**Why:** the tools are exposed once through a standard protocol, so any MCP client can use them (this agent, Claude Desktop, an IDE), and each side can be tested on its own: the tools in the MCP Inspector, the approval logic with a fake tool. It also gives two separate boundaries. The server decides what is possible at all: helpers like `read_file` are plain functions, not tools, so no client can ever call them. The agent decides, per call, whether a human approved it.

**How it fits together:**

```
graph_agent.py ── MultiServerMCPClient ── stdio ── jobs_server.py (FastMCP)
  approval + gate stay here                          list / read / score live here
```

- **Transport: stdio.** The agent starts the server as a child process (`sys.executable server/jobs_server.py`, with the path built from the repository root) and they exchange JSON-RPC messages over stdin/stdout. No port, no network, no auth to build, and the server lives exactly as long as the agent's session. If the tools had to be shared or deployed separately, the switch would be to Streamable HTTP plus authentication; the graph and gate would not change.
- **One session per run.** `client.session("Main")` keeps one server process for the whole run and `load_mcp_tools(session)` turns its tools into ordinary LangChain tools, so `bind_tools`, `ToolNode` and `need_approval` work unchanged. (`client.get_tools()` starts a fresh server process for every tool call: the log shows a `ListToolsRequest` before each `CallToolRequest`.) Everything that uses the tools runs inside the `async with` block.
- **Configuration at startup.** Which postings file the server reads is server configuration, not a tool parameter: the server reads `POSTINGS_FILE` once when it starts, and all three tools use it. A parameter would let the model choose the file, and a flag on one tool would leave the other two reading a different file.
- **Async.** MCP tools are async-only, so the gate awaits `tool_node.ainvoke(...)`, the graph runs with `ainvoke`, and `main()` is started once with `asyncio.run`. The graph is built in `build_graph(tools)` after the tools are loaded.
- **Testable gate.** `make_gate(tool_node)` builds the gate around any `ToolNode`. The agent passes the MCP tools; `test_interrupts.py` passes a fake `score_posting` with the same name and checks that an approved call runs once and a declined one never runs.

**What went wrong on the way:**

- **Version conflict.** In MCP SDK 2.x `FastMCP` was renamed `MCPServer`, but `langchain-mcp-adapters` (0.3.2) requires `mcp<2`. The SDK is pinned to 1.x.
- **Lost parameter descriptions.** LangChain's `@tool` reads a bare string in `Annotated[str, "..."]` as the description; the MCP SDK builds schemas with Pydantic, which ignores it. The descriptions use `Field(description=...)`, and the Inspector confirmed they reach the schema. That sentence is what tells the model to use ids exactly as listed.
- **A server that never started.** With `pass` instead of `mcp.run()`, `mcp run jobs_server.py` still worked (the CLI starts the server itself) but `python jobs_server.py` exited at once, so the Inspector showed no connection and no error.
- **Environment variables do not pass through.** The MCP SDK and the Inspector start the server with a small safe environment plus what is passed explicitly (`env` in the client config, `-e` in the Inspector), not the whole shell environment.

## Evaluation

`evals/run.py` measures the agent's decisions (qwen3 + profile + graph), not the scoring service.

**Cases.** `evals/cases.json` holds 42 labelled postings: my verdict, a one-line reason, one to three facts the reason must mention, and a fixed fake score. The labels were drafted with an LLM from my notes and I am still reviewing them. Some cases are traps taken from real failures: a junior AI role that should be applied to despite "Junior", a junior role that should be skipped because the work is low-code (not because of the title), two postings from one company that share the same intro text, and a strong content match that is ruled out by location.

**How a run works.**

- The run starts its own MCP client with `POSTINGS_FILE` pointing at `evals/postings_manual.json`, so the server reads frozen postings.
- `score_posting` is replaced, before the graph is built, by a fake with the same name and description that returns the case's `fake_score`. The scores are an input to the decision, so they are held fixed: with the real scorer, every run would get slightly different scores and a change in agreement could not be traced to the change that was made. It is also free and fast. (Same idea as `test_interrupts.py`: fake the edges, test the middle.)
- Every approval is answered "y" in code, so a run needs no keyboard.
- It prints each case (expected, got, reason), then agreement, false skips (apply cases the agent skipped, the costly mistake), must-mention hits and time.

**Baseline (24 Oct 2026), 20 cases:**

| Run | Agreement | False skips | Must-mention | Time |
|---|---|---|---|---|
| First baseline | 13/20 (65%) | 2 | 13/40 | 348 s |
| Temperature 0 | 12/20 (60%) | 3 | 10/40 | 356 s |

**What the misses show:**

- **Every false skip had a bare reason.** When the model wrote a real reason, the verdict was mostly right. When it wrote only the verdict word ("skip"), it was often wrong. The `reason` field needs a description that asks for the posting's deciding requirements.
- **All three "maybe" cases became skip.** My labels use maybe for a fallback (a strong non-AI or .NET role); the profile never says such roles are acceptable, so the model has no reason to choose maybe.
- **Single runs vary.** Two cases flipped between the two runs even at temperature 0, so a change is judged on more than one run.
- **One label is open.** The backend role asks for "proven experience scaling systems that serve millions of users". The model skips it on that requirement; my label says apply as a stretch. That is a labelling decision, not a bug.

**Limits of the eval:**

- 20 of 42 cases are scored. 20 cases have no saved posting text yet and 2 have only a one-line summary; a case judged without its text would measure title guessing, which is the bug the decide step fixed.
- The 20 lean towards apply (14 apply, 3 maybe, 3 skip), so "apply to everything" would already score 14/20. False skips and the skip cases matter more than overall agreement.
- The fake scorer returns only the number; the real one also returns matched and missing requirements, so the model has less evidence than in real use.
- Must-mention is a substring check, so it is a rough signal.

## Known limitations

- **Failed tools used to be invisible.** A 422 from the scoring service returned an error body with a 200-shaped `.json()`, so it reached the model looking like a result. The model noticed, said so in its reasoning, and then produced a confident ranking from scores it never received. Fixed with `raise_for_status()`; the lesson is that a tool failure the harness does not flag becomes a fabricated answer.
- **Local models need instructions spelled out.** Scoring was described as an option ("score the two or three strongest") and simply never happened. It had to become a requirement.
- **Tool descriptions are prompts.** A vague description means an unused tool.
- **One source.** Lever only. Most companies use other systems, and several have no readable API at all, so they are marked `manual` in the config and checked by hand.
- **`postings.json` holds one run's new postings**, so running the fetcher twice before scoring loses the batch.
- **Approval lives in the terminal.** The resume loop uses `input()`. A real deployment would put the question where the person already is (a ticket, a chat message) and resume when they answer.
- **No idempotency key on scoring.** The approval split stops the replay, but the scoring call itself is still not safe to repeat. An idempotency key would make an accidental second call harmless.
- **`input()` inside async code** blocks the event loop while waiting for an answer. Harmless here, because nothing else runs meanwhile; in a server with many users it would freeze everything.
- **Scoring has to be asked for.** With qwen3 the prompt has to say "score at least the two best postings", or the model skips the paid tool. Same lesson as above.
- **Declined calls are recognised by their text.** The decide step skips declined scores by matching the gate's message; marking them with `status="error"` in the gate would be cleaner.
- **Descriptions are sent as raw HTML**, which costs tokens without adding meaning.
