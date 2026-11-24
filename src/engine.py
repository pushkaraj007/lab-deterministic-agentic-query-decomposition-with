import re
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi
from src.models import Document, AgentState


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class MockRetriever:
    def __init__(self, corpus: List[Document]):
        self.corpus = corpus
        tokenized_corpus = [_tokenize(doc.text) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []
        
        # Get raw scores from BM25
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # We sort by scores descending and take top_k.
        # We remove the score > 0 filter because on very small/single-doc corpora,
        # BM25 can return negative or zero scores for valid matches.
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        return [self.corpus[i] for i in top_indices]


class Engine:
    """
    Orchestrates the multi-hop process.
    """
    def __init__(self, corpus: List[Document], max_hops: int = 3):
        self.retriever = MockRetriever(corpus)
        self.max_hops = max_hops

    def _validator(self, state: AgentState) -> bool:
        # Simulate LLM logic: check if raw_memory contains answer
        # In a real impl, this would be an LLM call.
        # Here, we look for keywords indicating "complete"
        full_text = " ".join([d.text for d in state.raw_memory]).lower()
        required = ["policy", "procedure", "limit"]
        return all(req in full_text for req in required)

    def _summarize(self, state: AgentState) -> str:
        # Mock summarization
        learned = [d.text[:50] for d in state.raw_memory]
        return f"Previously explored: {'; '.join(learned)}"

    def _reformulate_query(self, state: AgentState) -> str:
        # Simple heuristic: use original query on first hop, append context thereafter
        if state.hop_count <= 1:
            return state.original_query
        return f"{state.original_query} context: {state.compressed_history[:20]}"

    def run(self, query: str) -> AgentState:
        state = AgentState(query=query, original_query=query)

        for hop in range(self.max_hops):
            state.hop_count = hop + 1

            # 1. Reformulate
            current_query = self._reformulate_query(state)

            # 2. Retrieve
            new_docs = self.retriever.search(current_query)
            state.raw_memory.extend(new_docs)

            # 3. Compress History
            state.compressed_history = self._summarize(state)

            # 4. Check Sufficiency
            if self._validator(state):
                state.is_sufficient = True
                break

        return state