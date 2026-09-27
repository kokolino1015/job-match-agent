# Job posting agent

A tool-calling agent that fetches job postings, decides which are worth a closer look, and scores the promising ones against my CV. The loop runs locally on Ollama; scoring calls out to a separate service.

## How it works

```
fetch.py         Lever API -> normalised postings.json, deduplicated against seen.json
main.py          hand-written agent loop on Ollama: list -> read -> score -> ranked recommendation
graph_agent.py   the same agent rebuilt in LangGraph, with human approval before scoring
```

The agent has three tools:

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
python graph_agent.py   # or the LangGraph version, which asks before each score
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
START -> agent -> should_continue -> tools -> agent -> ... -> END
                        |
                        +-> END   (no tool calls left)
```

| Hand-written loop (`main.py`) | LangGraph (`graph_agent.py`) |
|---|---|
| `messages` list appended by hand | `MessagesState`, merged by the `add_messages` reducer |
| JSON schemas written by hand | `@tool` builds them from the name, docstring and `Annotated` hints |
| `run_tool` dispatcher and error prefix | `ToolNode(tools, handle_tool_errors=True)` |
| `while` loop and `if tool_calls` | conditional edge `should_continue` returning `"tools"` or `END` |
| `MAX_TURNS` guard | `recursion_limit` in the run config |
| no pause possible | `interrupt()` plus a checkpointer |

### Human in the loop

`score_posting` calls a paid API, so it asks first. `interrupt()` pauses the graph, and the checkpointer (`InMemorySaver`, keyed by `thread_id`) keeps the state. `invoke` returns a result with an `__interrupt__` key, and the run resumes with `Command(resume={interrupt_id: answer})`.

Approval fails closed. An empty answer counts as no, and the tool treats anything other than `"y"` as a refusal, telling the model to report the posting as not scored rather than estimate one.

### What testing it showed

- **Parallel calls pause one at a time.** When the model asks to score two postings in one turn, both calls run in the same `tools` step, and the first `interrupt` stops the whole step. The resume loop handles that by looping until no `__interrupt__` is left.
- **A resumed node reruns from the top.** So anything before `interrupt()` runs again on every resume. With two parallel score calls, the approved call gets replayed when the second one is answered, and its side effect (the POST to the scoring service) can run twice. The safer design is a separate approval node between `agent` and `tools`: one interrupt covering all pending calls, with no side effects in the tool itself.
- **The graph was tested with the model faked out.** A small graph containing only `ToolNode`, fed a hand-built `AIMessage` with two tool calls, reproduces the interrupt behaviour deterministically, without depending on what the model decides to do.

## Known limitations

- **Failed tools used to be invisible.** A 422 from the scoring service returned an error body with a 200-shaped `.json()`, so it reached the model looking like a result. The model noticed, said so in its reasoning, and then produced a confident ranking from scores it never received. Fixed with `raise_for_status()`; the lesson is that a tool failure the harness does not flag becomes a fabricated answer.
- **Local models need instructions spelled out.** Scoring was described as an option ("score the two or three strongest") and simply never happened. It had to become a requirement.
- **Tool descriptions are prompts.** A vague description means an unused tool.
- **One source.** Lever only. Most companies use other systems, and several have no readable API at all, so they are marked `manual` in the config and checked by hand.
- **`postings.json` holds one run's new postings**, so running the fetcher twice before scoring loses the batch.
- **No evaluation set.** Results were checked by comparing the agent's rankings against my own judgment on four postings I had already assessed. That caught a real bug (a junior-titled role dismissed on the title alone) but it is four data points, not a benchmark.