import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.graph_agent import build_graph, read_file

POSTINGS = ROOT / "evals" / "postings_manual.json"
QUESTION = "Which of my saved postings should I look at, and why? Score at least the two best postings."


class ScoreArgs(BaseModel):
    id: str = Field(description="The posting id, exactly as returned by list_posting")


def make_fake_scorer(cases_by_id):
    async def score_posting(id: str) -> str:
        return json.dumps({"score": cases_by_id[id]["fake_score"]})

    return StructuredTool.from_function(
        coroutine=score_posting,
        name="score_posting",
        description="Run a detailed analysis of the CV against a posting. Returns a match score.",
        args_schema=ScoreArgs,
    )


async def run_agent(cases_by_id):
    client = MultiServerMCPClient({
        "Main": {
            "command": sys.executable,
            "args": [str(ROOT / "server" / "jobs_server.py")],
            "transport": "stdio",
            "env": {"POSTINGS_FILE": str(POSTINGS)},
        }
    })
    async with client.session("Main") as session:
        tools = await load_mcp_tools(session)
        tools = [make_fake_scorer(cases_by_id) if t.name == "score_posting" else t for t in tools]
        graph = build_graph(tools)
        config = {"configurable": {"thread_id": "eval"}, "recursion_limit": 60}

        result = await graph.ainvoke(
            {"messages": [SystemMessage(read_file(str(ROOT / "data" / "profile.txt"))), HumanMessage(QUESTION)]},
            config)
        while "__interrupt__" in result:
            answers = {i["id"]: "y" for i in result["__interrupt__"][0].value}
            result = await graph.ainvoke(Command(resume=answers), config)

    return {d.posting_id: d for d in result["result"].decisions}


def main():
    with open("evals/cases.json") as f:
        cases_by_id = {c["id"]: c for c in json.load(f)}
    with open(POSTINGS) as f:
        postings = {p["id"]: p for p in json.load(f)}

    start = time.perf_counter()
    decisions = asyncio.run(run_agent(cases_by_id))
    seconds = time.perf_counter() - start

    cases = [c for c in cases_by_id.values() if c["id"] in postings and "summary only" not in postings[c["id"]]["description"]]

    agree, false_skips, hits, total = 0, [], 0, 0
    for case in cases:
        d = decisions.get(case["id"])
        got = d.verdict if d else "none"
        reason = d.reason if d else ""
        agree += got == case["verdict"]
        if case["verdict"] == "apply" and got == "skip":
            false_skips.append(case["title"])
        hits += sum(m.lower() in reason.lower() for m in case["must_mention"])
        total += len(case["must_mention"])
        mark = "ok  " if got == case["verdict"] else "MISS"
        print(f"{mark} expected {case['verdict']:5} got {got:5}  {case['title'][:45]:45}  {reason[:90]}")

    print()
    print(f"Agreement:    {agree}/{len(cases)} ({agree / len(cases):.0%})")
    print(f"False skips:  {len(false_skips)} {false_skips}")
    print(f"Must-mention: {hits}/{total}")
    print(f"Time:         {seconds:.0f} s")


if __name__ == "__main__":
    main()
