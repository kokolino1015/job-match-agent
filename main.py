import json

import ollama
import requests

from config import MAX_TURNS, OLLAMA_MODEL, OLLAMA_MAX_TOKENS


def list_posting():
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


def read_posting(id: str):
    with open("postings.json") as f:
        for posting in json.load(f):
            if posting["id"] == id:
                return posting["description"]

    raise ValueError(f"Posting with id {id} not found")


def read_file(fileName: str):
    with open(fileName, 'r') as f:
        return f.read()


def score_posting(id: str):
    job_listing = read_posting(id)
    cv = read_file("cv.txt")
    provider = "anthropic"

    return requests.post("http://localhost:8000/analyze_structured/", json={
        "cv_text": cv,
        "provider": provider,
        "job_listing": job_listing
    }).json()


TOOLS = {"list_posting": list_posting, "read_posting": read_posting, "score_posting": score_posting}

ID = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "posting id"},
    },
    "required": ["id"],
}

EMPTY_INPUT = {"type": "object", "properties": {}}

TOOL_SCHEMAS = [
    {"type": "function",
     "function": {"name": "read_posting", "description": "Read posting by the given id.", "parameters": ID}},
    {"type": "function",
     "function": {"name": "list_posting", "description": "Lists all postings by title, company, id.",
                  "parameters": EMPTY_INPUT}},
    {"type": "function",
     "function": {"name": "score_posting", "description": "Run a detailed analysis of the CV against a posting. Returns a match score and a breakdown of which requirements are met and which are not. Use this to confirm fit on shortlisted postings before recommending them.", "parameters": ID}},
]


def run_tool(name, tool_input):
    func = TOOLS.get(name)
    if func is None:
        return f"Unknown tool: {name}", True
    try:
        return str(func(**tool_input)), False
    except Exception as e:
        return f"{type(e).__name__}: {e}", True


def run_agent(question):
    messages = [
        {"role": "system", "content": read_file("profile.txt")},
        {"role": "user", "content": question}
    ]

    for _ in range(MAX_TURNS):
        response = ollama.chat(
            model=OLLAMA_MODEL,
            tools=TOOL_SCHEMAS,
            messages=messages,
            think=False,
            options={"num_ctx": 32768, "num_predict": OLLAMA_MAX_TOKENS}
        )

        messages.append(response.message)

        if response.done_reason != "stop":
            raise RuntimeError(f"Model stopped early: {response.done_reason}")

        if response.message.tool_calls:
            for call in response.message.tool_calls:
                content, is_error = run_tool(call.function.name, call.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_name": call.function.name,
                    "content": f"ERROR: {content}" if is_error else content,
                })
        else:
            return response.message.content

    raise RuntimeError(f"Agent did not finish within {MAX_TURNS} turns")


if __name__ == "__main__":
    print(run_agent("Which of my saved postings should I look at, and why?"))
