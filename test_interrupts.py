"""Shows what reruns when two score_posting calls in one step both pause.

Before running, add two prints to score_posting in graph_agent.py:
    print(f"ASKING {id}")    as the first line, before interrupt(...)
    print(f"POSTING {id}")   right before requests.post(...)

Run from the project folder:  python test_interrupts.py
The model is not used, and the analyzer does not need to be running.
"""
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from graph_agent import tools

# A graph with only the tools step: START -> tools -> END
builder = StateGraph(MessagesState)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
builder.add_edge(START, "tools")
builder.add_edge("tools", END)
test_graph = builder.compile(checkpointer=InMemorySaver())

# A hand-made model message asking for two scores in one turn
fake = AIMessage(content="", tool_calls=[
    {"name": "score_posting", "args": {"id": "manual:ai-automation-engineer-sofia"}, "id": "call_a"},
    {"name": "score_posting", "args": {"id": "manual:pwc-junior-ai-developer"}, "id": "call_b"},
])

cfg = {"configurable": {"thread_id": "test-rerun"}}


def answer_for(question: str) -> str:
    """Scripted answers: yes to the automation posting, no to PwC."""
    return "y" if "ai-automation" in question else "n"


print("--- invoke 1")
result = test_graph.invoke({"messages": [fake]}, cfg)

invoke_number = 1
while "__interrupt__" in result:
    answers = {}
    for intr in result["__interrupt__"]:
        answers[intr.id] = answer_for(intr.value)
        print(f"    paused on: {intr.value!r}  -> answering {answers[intr.id]}")

    invoke_number += 1
    print(f"--- invoke {invoke_number}")
    result = test_graph.invoke(Command(resume=answers), cfg)

print("--- finished. Tool results:")
for m in result["messages"][1:]:
    print(f"    {m.tool_call_id}: {str(m.content)[:100]}")