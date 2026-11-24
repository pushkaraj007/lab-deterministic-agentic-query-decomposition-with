"""Headless CPU runner executed by GitHub Actions (workflow_dispatch, timeout 10m).
No secrets are stored in this repository. Writes structured metrics.json & environment.json.
"""
import io
import json
import platform
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPTURED_METRICS = []
CAPTURED_ENV = {
    "hardware": f"CPU ({platform.machine()})",
    "runtime": f"GitHub Actions ({platform.system()} {platform.release()})",
    "pythonVersion": platform.python_version(),
    "datasetSource": "Synthetic reproducible benchmark (fixed seed = 42)",
    "databaseSource": "None (in-memory / DuckDB)",
    "randomSeed": "42",
}

def log_lab_environment(extra=None):
    if isinstance(extra, dict):
        CAPTURED_ENV.update(extra)
    print("[LAB_ENV_JSON]", json.dumps(CAPTURED_ENV))
    return CAPTURED_ENV

def log_lab_result(metrics):
    if isinstance(metrics, dict):
        print("[LAB_RESULT_JSON]", json.dumps(metrics))

def main():
    buf = io.StringIO()
    entry_path = ROOT / "main.py"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if entry_path.exists():
        import runpy
        with redirect_stdout(buf):
            runpy.run_path(str(entry_path), run_name="__main__")
    else:
        script_source = "# --- Cell 1: src/models.py ---\nfrom pydantic import BaseModel, Field\nfrom typing import List, Optional\n\nclass Document(BaseModel):\n    doc_id: str\n    text: str\n    metadata: dict = Field(default_factory=dict)\n\nclass AgentState(BaseModel):\n    query: str\n    original_query: str\n    hop_count: int = 0\n    raw_memory: List[Document] = Field(default_factory=list)\n    compressed_history: str = \"\"\n    is_sufficient: bool = False\n\nclass SearchResult(BaseModel):\n    query: str\n    results: List[Document]\n\n# --- Cell 2: src/engine.py ---\nimport re\nimport numpy as np\nfrom typing import List\nfrom rank_bm25 import BM25Okapi\nfrom src.models import Document, AgentState\n\n\ndef _tokenize(text: str) -> List[str]:\n    return re.findall(r\"\\w+\", text.lower())\n\n\nclass MockRetriever:\n    def __init__(self, corpus: List[Document]):\n        self.corpus = corpus\n        tokenized_corpus = [_tokenize(doc.text) for doc in corpus]\n        self.bm25 = BM25Okapi(tokenized_corpus)\n\n    def search(self, query: str, top_k: int = 3) -> List[Document]:\n        tokenized_query = _tokenize(query)\n        if not tokenized_query:\n            return []\n        doc_scores = self.bm25.get_scores(tokenized_query)\n        top_indices = np.argsort(doc_scores)[::-1][:top_k]\n        return [self.corpus[i] for i in top_indices if doc_scores[i] > 0]\n\n\nclass Engine:\n    \"\"\"\n    Orchestrates the multi-hop process.\n    \"\"\"\n    def __init__(self, corpus: List[Document], max_hops: int = 3):\n        self.retriever = MockRetriever(corpus)\n        self.max_hops = max_hops\n\n    def _validator(self, state: AgentState) -> bool:\n        # Simulate LLM logic: check if raw_memory contains answer\n        # In a real impl, this would be an LLM call.\n        # Here, we look for keywords indicating \"complete\"\n        full_text = \" \".join([d.text for d in state.raw_memory]).lower()\n        required = [\"policy\", \"procedure\", \"limit\"]\n        return all(req in full_text for req in required)\n\n    def _summarize(self, state: AgentState) -> str:\n        # Mock summarization\n        learned = [d.text[:50] for d in state.raw_memory]\n        return f\"Previously explored: {'; '.join(learned)}\"\n\n    def _reformulate_query(self, state: AgentState) -> str:\n        # Simple heuristic: use original query on first hop, append context thereafter\n        if state.hop_count <= 1:\n            return state.original_query\n        return f\"{state.original_query} context: {state.compressed_history[:20]}\"\n\n    def run(self, query: str) -> AgentState:\n        state = AgentState(query=query, original_query=query)\n\n        for hop in range(self.max_hops):\n            state.hop_count = hop + 1\n\n            # 1. Reformulate\n            current_query = self._reformulate_query(state)\n\n            # 2. Retrieve\n            new_docs = self.retriever.search(current_query)\n            state.raw_memory.extend(new_docs)\n\n            # 3. Compress History\n            state.compressed_history = self._summarize(state)\n\n            # 4. Check Sufficiency\n            if self._validator(state):\n                state.is_sufficient = True\n                break\n\n        return state\n\n# --- Cell 3: tests/test_core.py ---\nimport os\nimport sys\n\n# Ensure src is in the path for direct execution\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), \"../\")))\n\nfrom src.models import Document\nfrom src.engine import Engine\n\n\ndef test_hop_limit_enforcement():\n    corpus = [Document(doc_id=\"1\", text=\"some random text\")]\n    engine = Engine(corpus, max_hops=2)\n    state = engine.run(\"test query\")\n    assert state.hop_count <= 2\n\n\ndef test_validator_logic():\n    corpus = [\n        Document(doc_id=\"1\", text=\"The policy document defines the rules.\"),\n        Document(doc_id=\"2\", text=\"The procedure is strict.\"),\n        Document(doc_id=\"3\", text=\"There is a limit to the hops.\"),\n    ]\n    engine = Engine(corpus, max_hops=3)\n    state = engine.run(\"What is the policy procedure limit?\")\n    assert state.is_sufficient is True\n\n\ndef test_cycle_avoidance_structure():\n    # Ensures the loop structure is deterministic and terminates at max_hops\n    corpus = [Document(doc_id=\"1\", text=\"trigger data\")]\n    engine = Engine(corpus, max_hops=3)\n    state = engine.run(\"trigger\")\n    assert len(state.raw_memory) > 0\n    assert state.hop_count == 3\n    assert state.is_sufficient is False\n\n# --- Cell 4: main.py ---\nimport json\nfrom src.models import Document\nfrom src.engine import Engine\n\ndef main():\n    # Synthetic dataset\n    corpus = [\n        Document(doc_id=\"1\", text=\"Company policy requires signed approvals.\"),\n        Document(doc_id=\"2\", text=\"The procedure for submittal involves a web form.\"),\n        Document(doc_id=\"3\", text=\"There is a hard limit of 5 days for review.\"),\n        Document(doc_id=\"4\", text=\"Irrelevant noise about coffee breaks.\")\n    ]\n\n    engine = Engine(corpus, max_hops=3)\n    \n    # Run Benchmark\n    query = \"What is the policy procedure limit?\"\n    result = engine.run(query)\n\n    # Calculate metrics\n    metrics = {\n        \"success\": result.is_sufficient,\n        \"hops_taken\": result.hop_count,\n        \"raw_memory_size\": len(result.raw_memory),\n        \"recall_met\": result.is_sufficient # Simplified proxy\n    }\n    \n    print(f\"Orchestration complete. Hops: {result.hop_count}, Sufficient: {result.is_sufficient}\")\n    print(\"[LAB_RESULT_JSON]\", json.dumps(metrics))\n\nif __name__ == \"__main__\":\n    main()\n\n# --- Cell 5: src/__init__.py ---\n# Mark src as a package\n\n# --- Cell 6: tests/__init__.py ---\n# Mark tests as a package\n\n# --- Cell 7: tests/conftest.py ---\nimport os\nimport sys\n\n# Ensure repository root is on sys.path for pytest collection in CI\nROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\"))\nif ROOT_DIR not in sys.path:\n    sys.path.insert(0, ROOT_DIR)"
        ns = {
            "__name__": "__main__",
            "log_lab_environment": log_lab_environment,
            "log_lab_result": log_lab_result,
        }
        with redirect_stdout(buf):
            exec(compile(script_source, "lab_cells.py", "exec"), ns)

    raw_output = buf.getvalue()
    if raw_output:
        print(raw_output, end="" if raw_output.endswith("\n") else "\n")

    now_iso = datetime.now(timezone.utc).isoformat()
    for line in raw_output.splitlines():
        if "[LAB_RESULT_JSON]" in line:
            raw = line.split("[LAB_RESULT_JSON]", 1)[1].strip()
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        CAPTURED_METRICS.append({
                            "key": str(k)[:64],
                            "value": v if isinstance(v, (int, float)) else str(v)[:120],
                            "unit": "",
                            "baseline": None,
                            "delta": "",
                            "source": "actions",
                            "capturedAt": now_iso,
                        })
            except Exception:
                pass
        elif "[LAB_ENV_JSON]" in line:
            raw = line.split("[LAB_ENV_JSON]", 1)[1].strip()
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    CAPTURED_ENV.update(obj)
            except Exception:
                pass

    metrics_path = ROOT / "metrics.json"
    existing = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    if CAPTURED_METRICS:
        existing["metrics"] = CAPTURED_METRICS
    elif "metrics" not in existing:
        existing["metrics"] = []
    existing["labId"] = "lab_mwv7zvsn_7bkm"
    existing["title"] = "Deterministic Agentic Query Decomposition with Hop-Limit Termination"
    existing["generatedAt"] = now_iso
    metrics_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

    env_path = ROOT / "environment.json"
    env_path.write_text(json.dumps(CAPTURED_ENV, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(CAPTURED_METRICS)} metric(s) to metrics.json")

if __name__ == "__main__":
    main()
