from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

import graph_agent
from graph_agent import gate, approval, State

builder = StateGraph(State)
builder.add_node("gate", gate)
builder.add_node("approval", approval)
builder.add_edge(START, "approval")
builder.add_edge("approval", "gate")
builder.add_edge("gate", END)
test_graph = builder.compile(checkpointer=InMemorySaver())
posts = []

class FakeResp:
    def raise_for_status(self): pass

    def json(self): return {"score": 80}


def fake_post(url, json):
    posts.append(json["job_listing"][:40])   # remember which posting was scored
    return FakeResp()

graph_agent.requests.post = fake_post

fake = AIMessage(content="", tool_calls=[
    {"name": "score_posting", "args": {"id": "manual:ai-automation-engineer-sofia"}, "id": "call_a"},
    {"name": "score_posting", "args": {"id": "manual:pwc-junior-ai-developer"}, "id": "call_b"},
])

cfg = {"configurable": {"thread_id": "test-rerun"}}


def answer_for(question: str) -> str:
    return "y" if "ai-automation" in question else "n"


print("--- invoke 1")
result = test_graph.invoke({"messages": [fake]}, cfg)

invoke_number = 1
while "__interrupt__" in result:
    answers = {}
    for intr in result["__interrupt__"][0].value:
        answers[intr["id"]] = answer_for(intr["message"])
        print(f"    paused on: {intr["message"]!r}  -> answering {answers[intr["id"]]}")

    invoke_number += 1
    print(f"--- invoke {invoke_number}")
    result = test_graph.invoke(Command(resume=answers), cfg)

print("--- finished. Tool results:")
for m in result["messages"][1:]:
    print(f"    {m.tool_call_id}: {str(m.content)[:100]}")


results = {m.tool_call_id: str(m.content) for m in result["messages"][1:]}

assert len(posts) == 1, f"expected 1 POST, got {len(posts)}: {posts}"
assert "declined" in results["call_b"]
assert "80" in results["call_a"]
print("OK: approved call ran once, declined call never ran")