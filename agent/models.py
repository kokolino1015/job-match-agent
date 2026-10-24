from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel


class Verdict(BaseModel):
    reason: str
    verdict: Literal["apply", "maybe", "skip"]


class Decision(BaseModel):
    posting_id: str
    reason: str
    verdict: Literal["apply", "maybe", "skip"]


class AgentResult(BaseModel):
    decisions: list[Decision]


class State(MessagesState):
    approvals: dict
    result: AgentResult | None
