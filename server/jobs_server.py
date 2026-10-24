import json
import os
from typing import Annotated

import requests
from mcp.server import FastMCP
from pydantic import Field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

POSTINGS_FILE = os.environ.get("POSTINGS_FILE", str(DATA / "postings.json"))

mcp = FastMCP("Main")


@mcp.tool()
def list_posting():
    """Lists all postings by title, company, id."""
    result = []
    with open(POSTINGS_FILE) as f:
        for posting in json.load(f):
            result.append({
                "id": posting["id"],
                "title": posting["title"],
                "location": posting["location"],
                "company": posting["slug"]
            })

    return result


def load_description(posting_id: str) -> str:
    with open(POSTINGS_FILE) as f:
        for posting in json.load(f):
            if posting["id"] == posting_id:
                return posting["description"]
    raise ValueError(f"Posting with id {posting_id} not found")


@mcp.tool()
def read_posting(id: Annotated[str, Field(description="The posting id, exactly as returned by list_posting")]) -> str:
    """Read the full description of one job posting by its id."""
    return load_description(id)


def read_file(fileName: str):
    with open(fileName, 'r') as f:
        return f.read()


@mcp.tool()
def score_posting(id: Annotated[str, Field(description="The posting id, exactly as returned by list_posting")]):
    """Run a detailed analysis of the CV against a posting. Returns a match score and a breakdown of which requirements are met and which are not. Use this to confirm fit on shortlisted postings before recommending them."""
    job_listing = load_description(id)
    cv = read_file(str(DATA / "cv.txt"))
    provider = "anthropic"

    response = requests.post("http://localhost:8000/analyze_structured/", json={
        "cv_text": cv,
        "provider": provider,
        "job_listing": job_listing
    })

    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
