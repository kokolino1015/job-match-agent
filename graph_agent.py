import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

client = MultiServerMCPClient({
    "Main": {
        "command": "python",
        "args": ["jobs_server.py"],
        "transport": "stdio",
    }
})


class State(MessagesState):
    decisions: dict


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "approval"
    return END


def approval(state: State):
    risky = [tool_call for tool_call in state["messages"][-1].tool_calls if tool_call["name"] in need_approval]

    if not risky:
        return {"decisions": {}}

    answers = interrupt(
        [{"message": f"Score posting {tool_call['args']['id']}?", "id": tool_call["id"]} for tool_call in risky])

    return {"decisions": answers}


need_approval = {"score_posting"}


def read_file(fileName: str):
    with open(fileName, 'r') as f:
        return f.read()


def build_graph(tools):
    tool_node = ToolNode(tools, handle_tool_errors=True)
    llm_with_tools = ChatOllama(model="qwen3:30b", num_ctx=32768).bind_tools(tools)

    def agent(state: MessagesState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    async def gate(state: State):
        last = state["messages"][-1]
        decisions = state.get("decisions", {})

        approved, declined = [], []
        for call in last.tool_calls:
            if call["name"] in need_approval and decisions.get(call["id"]) != "y":
                declined.append(call)
            else:
                approved.append(call)

        messages = [ToolMessage(
            content=f"The user declined {call['name']}. Do not estimate a score; report this posting as not scored.",
            tool_call_id=call["id"]) for call in declined]

        if approved:
            only_approved = last.model_copy(update={"tool_calls": approved})
            messages += (await tool_node.ainvoke({"messages": [only_approved]}))["messages"]

        return {"messages": messages}

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("gate", gate)
    builder.add_node("approval", approval)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["approval", END])
    builder.add_edge("gate", "agent")
    builder.add_edge("approval", "gate")
    return builder.compile(checkpointer=InMemorySaver())


async def main():
    async with client.session("Main") as session:
        tools = await load_mcp_tools(session)
        graph = build_graph(tools)
        config = {"configurable": {"thread_id": "run-1"}, "recursion_limit": 20}

        result = await graph.ainvoke(
            {"messages": [SystemMessage(read_file("profile.txt")),
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

        for m in result["messages"]:
            m.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
