# Job posting agent

A tool-calling agent that fetches job postings, decides which are worth a closer look, and scores the promising ones against my CV. The loop runs locally on Ollama; scoring calls out to a separate service.

## How it works

```
fetch.py      Lever API -> normalised postings.json, deduplicated against seen.json
main.py       agent loop: list -> read -> score -> ranked recommendation
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
python main.py      # run the agent
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

## Known limitations

- **Failed tools used to be invisible.** A 422 from the scoring service returned an error body with a 200-shaped `.json()`, so it reached the model looking like a result. The model noticed, said so in its reasoning, and then produced a confident ranking from scores it never received. Fixed with `raise_for_status()`; the lesson is that a tool failure the harness does not flag becomes a fabricated answer.
- **Local models need instructions spelled out.** Scoring was described as an option ("score the two or three strongest") and simply never happened. It had to become a requirement.
- **Tool descriptions are prompts.** A vague description means an unused tool.
- **One source.** Lever only. Most companies use other systems, and several have no readable API at all, so they are marked `manual` in the config and checked by hand.
- **`postings.json` holds one run's new postings**, so running the fetcher twice before scoring loses the batch.
- **No evaluation set.** Results were checked by comparing the agent's rankings against my own judgment on four postings I had already assessed. That caught a real bug (a junior-titled role dismissed on the title alone) but it is four data points, not a benchmark.