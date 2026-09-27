import json

from langchain_core.tools import tool
from typing import Annotated
import requests
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

@tool
def list_posting():
    """Lists all postings by title, company, id."""
    result = []
    with open("postings.json") as f:
        for posting in json.load(f):
            result.append({
                "id": posting["id"],
                "title": posting["title"],
                "location": posting["location"],
                "company": posting["slug"]
            })

    return result


def load_description(posting_id: str) -> str:
    with open("postings.json") as f:
        for posting in json.load(f):
            if posting["id"] == posting_id:
                return posting["description"]
    raise ValueError(f"Posting with id {posting_id} not found")


@tool
def read_posting(id: Annotated[str, "The posting id, exactly as returned by list_posting"]) -> str:
    """Read the full description of one job posting by its id."""
    return load_description(id)


def read_file(fileName: str):
    with open(fileName, 'r') as f:
        return f.read()


@tool
def score_posting(id: Annotated[str, "The posting id, exactly as returned by list_posting"]):
    """Run a detailed analysis of the CV against a posting. Returns a match score and a breakdown of which requirements are met and which are not. Use this to confirm fit on shortlisted postings before recommending them."""
    answer = interrupt(f"Score posting {id}?")
    if answer != "y":
        return "The user declined to score this posting. Do not estimate a score; report it as not scored."
    job_listing = load_description(id)
    cv = read_file("cv.txt")
    provider = "anthropic"

    response = requests.post("http://localhost:8000/analyze_structured/", json={
        "cv_text": cv,
        "provider": provider,
        "job_listing": job_listing
    })

    response.raise_for_status()
    return response.json()


def agent(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END


tools = [list_posting, read_posting, score_posting]
llm_with_tools = ChatOllama(model="qwen3:30b", num_ctx=32768).bind_tools(tools)

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "run-1"}, "recursion_limit": 20}

    result = graph.invoke(
        {"messages": [SystemMessage(read_file("profile.txt")),
                      HumanMessage("Which of my saved postings should I look at, and why?")]},
        config,
    )

    while "__interrupt__" in result:
        answers = {}
        for intr in result["__interrupt__"]:
            answer = ""
            while answer not in ("y", "n"):
                answer = input(f"{intr.value} [y/N] ").strip().lower() or "n"
            answers[intr.id] = answer
        result = graph.invoke(Command(resume=answers), config)

    for m in result["messages"]:
        m.pretty_print()