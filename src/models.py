from pydantic import BaseModel, Field
from typing import List, Optional

class Document(BaseModel):
    doc_id: str
    text: str
    metadata: dict = Field(default_factory=dict)

class AgentState(BaseModel):
    query: str
    original_query: str
    hop_count: int = 0
    raw_memory: List[Document] = Field(default_factory=list)
    compressed_history: str = ""
    is_sufficient: bool = False

class SearchResult(BaseModel):
    query: str
    results: List[Document]