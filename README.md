# Deterministic Agentic Query Decomposition with Hop-Limit Termination

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pushkaraj007/lab-deterministic-agentic-query-decomposition-with/blob/main/notebook.ipynb)

## Problem Statement
Naive multi-hop RAG agents enter infinite reformulation loops when corpora lack the requested evidence, burning tokens and timing out. Single-shot retrieval fails when answers require bridging two distinct policy documents.

**Target Success Metric:** Multi-hop Recall@10 >= 0.85 with <= 3.0 average hops and 0% infinite loops

## Reproducibility & Environment
- **Hardware:** GitHub Actions Ubuntu CPU (2-core) / Google Colab T4 GPU
- **Runtime / Python:** Python 3.11 (Python 3.11)
- **Dataset Source:** Synthetic reproducible benchmark (fixed seed = 42)
- **Database Source:** None (in-memory / DuckDB)
- **Random Seed:** `42`

### Quick Start
```bash
pip install -r requirements.txt
pytest tests/test_solution.py
```

## Measured Results
_Run the notebook in GitHub Actions CPU or Google Colab and log metrics via `log_lab_result({...})` to populate this table._
