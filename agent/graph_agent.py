import asyncio
import json
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from agent.models import State, AgentResult, Decision, Verdict

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

need_approval = {"score_posting"}

client = MultiServerMCPClient({
    "Main": {
        "command": sys.executable,
        "args": [str(ROOT / "server" / "jobs_server.py")],
        "transport": "stdio",
    }
})


def tool_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c["text"] if isinstance(c, dict) else str(c) for c in content)


def tool_json(content):
    if isinstance(content, str):
        return json.loads(content)
    parts = [json.loads(tool_text([c])) for c in content]
    return parts[0] if len(parts) == 1 else parts


def read_file(fileName: str):
    with open(fileName, 'r') as f:
        return f.read()


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "approval"
    return "decide"


def approval(state: State):
    risky = [tool_call for tool_call in state["messages"][-1].tool_calls if tool_call["name"] in need_approval]

    if not risky:
        return {"approvals": {}}

    answers = interrupt(
        [{"message": f"Score posting {tool_call['args']['id']}?", "id": tool_call["id"]} for tool_call in risky])

    return {"approvals": answers}


def build_graph(tools):
    tool_node = ToolNode(tools, handle_tool_errors=True)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = ChatOllama(model="qwen3:30b", num_ctx=32768, temperature=0).bind_tools(tools)
    llm_decide = (ChatOllama(model="qwen3:30b", num_ctx=32768, temperature=0, num_predict=4096)
        .with_structured_output(Verdict, include_raw=True))

    async def agent(state: MessagesState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    def scores_from_history(messages):
        call_to_posting = {}
        for m in messages:
            if isinstance(m, AIMessage):
                for call in m.tool_calls:
                    if call["name"] == "score_posting":
                        call_to_posting[call["id"]] = call["args"]["id"]

        scored = {}
        for m in messages:
            if isinstance(m, ToolMessage) and m.tool_call_id in call_to_posting.keys():
                if m.status == "error":
                    continue
                scored[call_to_posting[m.tool_call_id]] = m.text

        return scored

    async def decide(state: State):
        scores = scores_from_history(state["messages"])
        profile = read_file(str(DATA / "profile.txt"))
        postings = tool_json(await tools_by_name["list_posting"].ainvoke({}))
        decisions = []
        for p in postings:
            text = tool_text(await tools_by_name["read_posting"].ainvoke({"id": p["id"]}))
            score = scores.get(p["id"], "Not scored.")
            v = await llm_decide.ainvoke([
                SystemMessage(profile),
                HumanMessage(
                    f"Title: {p['title']}\nCompany: {p['company']}\nLocation: {p['location']}\n\n"
                    f"Description:\n{text}\n\nCV match analysis:\n{score}\n\n"
                    "Give your verdict (apply / maybe / skip) and a one-line reason based on this posting.")
            ])
            parsed = v["parsed"]
            if parsed is None:
                print(f"NO VERDICT {p['id']}: done_reason={v['raw'].response_metadata.get('done_reason')} "
                      f"content={v['raw'].content[:200]!r}")
                continue
            decisions.append(Decision(posting_id=p["id"], verdict=parsed.verdict, reason=parsed.reason))
        return {"result": AgentResult(decisions=decisions)}

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("gate", make_gate(tool_node))
    builder.add_node("approval", approval)
    builder.add_node("decide", decide)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["approval", "decide"])
    builder.add_edge("gate", "agent")
    builder.add_edge("approval", "gate")
    builder.add_edge("decide", END)
    return builder.compile(checkpointer=InMemorySaver())


async def main():
    async with client.session("Main") as session:
        tools = await load_mcp_tools(session)
        graph = build_graph(tools)
        config = {"configurable": {"thread_id": "run-1"}, "recursion_limit": 20}

        result = await graph.ainvoke(
            {"messages": [SystemMessage(read_file(str(DATA / "profile.txt"))),
                          HumanMessage(
                              "Which of my saved postings should I look at, and why? Score at least the two best postings.")]},
            config,
        )

        while "__interrupt__" in result:
            answers = {}
            for intr in result["__interrupt__"][0].value:
                answer = ""
                while answer not in ("y", "n"):
                    answer = input(f"{intr['message']} [y/N] ").strip().lower() or "n"
                answers[intr["id"]] = answer
            result = await graph.ainvoke(Command(resume=answers), config)

        for d in result["result"].decisions:
            print(f"{d.verdict:5}  {d.posting_id}  {d.reason}")


def make_gate(tool_node):
    async def gate(state: State):
        last = state["messages"][-1]
        approvals = state.get("approvals", {})

        approved, declined = [], []
        for call in last.tool_calls:
            if call["name"] in need_approval and approvals.get(call["id"]) != "y":
                declined.append(call)
            else:
                approved.append(call)

        messages = [ToolMessage(
            content=f"The user declined {call['name']}. Do not estimate a score; report this posting as not scored.",
            status="error",
            name=call["name"],
            tool_call_id=call["id"]) for call in declined]

        if approved:
            only_approved = last.model_copy(update={"tool_calls": approved})
            messages += (await tool_node.ainvoke({"messages": [only_approved]}))["messages"]

        return {"messages": messages}

    return gate


if __name__ == "__main__":
    asyncio.run(main())
