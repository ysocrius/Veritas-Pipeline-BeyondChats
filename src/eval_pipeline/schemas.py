from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: str
    id: Optional[str] = None
    timestamp: Optional[float] = None

class Conversation(BaseModel):
    id: str
    messages: List[Message]

class ContextChunk(BaseModel):
    text: str
    vector: List[float] = Field(default_factory=list)
    score: Optional[float] = None

# Context is keyed by the ID of the USER message it retrieves for.
class ContextData(BaseModel):
    entries: Dict[str, List[ContextChunk]]

class EvalInput(BaseModel):
    conversation: Conversation
    context: ContextData
