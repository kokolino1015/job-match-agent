import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from graph_agent import approval, State, make_gate

calls = []

@tool
def score_posting(id: str) -> dict:
    """Fake scorer for the test."""
    calls.append(id)
    return {"score": 80}

builder = StateGraph(State)
builder.add_node("gate", make_gate(ToolNode([score_posting])))
builder.add_node("approval", approval)
builder.add_edge(START, "approval")
builder.add_edge("approval", "gate")
builder.add_edge("gate", END)
test_graph = builder.compile(checkpointer=InMemorySaver())

fake = AIMessage(content="", tool_calls=[
    {"name": "score_posting", "args": {"id": "manual:ai-automation-engineer-sofia"}, "id": "call_a"},
    {"name": "score_posting", "args": {"id": "manual:pwc-junior-ai-developer"}, "id": "call_b"},
])

cfg = {"configurable": {"thread_id": "test-rerun"}}


def answer_for(question: str) -> str:
    return "y" if "ai-automation" in question else "n"


print("--- invoke 1")

async def main():
    result = await test_graph.ainvoke({"messages": [fake]}, cfg)
    invoke_number = 1
    while "__interrupt__" in result:
        answers = {}
        for intr in result["__interrupt__"][0].value:
            answers[intr["id"]] = answer_for(intr["message"])
            print(f"    paused on: {intr["message"]!r}  -> answering {answers[intr["id"]]}")

        invoke_number += 1
        print(f"--- invoke {invoke_number}")
        result = await test_graph.ainvoke(Command(resume=answers), cfg)

    print("--- finished. Tool results:")
    for m in result["messages"][1:]:
        print(f"    {m.tool_call_id}: {str(m.content)[:100]}")


    results = {m.tool_call_id: str(m.content) for m in result["messages"][1:]}

    assert len(calls) == 1, f"expected 1 POST, got {len(calls)}: {calls}"
    assert "declined" in results["call_b"]
    assert "80" in results["call_a"]
    print("OK: approved call ran once, declined call never ran")

if __name__ == "__main__":
    asyncio.run(main())