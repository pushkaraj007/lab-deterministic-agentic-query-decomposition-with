import os
import sys

# Ensure src is in the path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.models import Document
from src.engine import Engine


def test_hop_limit_enforcement():
    corpus = [Document(doc_id="1", text="some random text")]
    engine = Engine(corpus, max_hops=2)
    state = engine.run("test query")
    assert state.hop_count <= 2


def test_validator_logic():
    corpus = [
        Document(doc_id="1", text="The policy document defines the rules."),
        Document(doc_id="2", text="The procedure is strict."),
        Document(doc_id="3", text="There is a limit to the hops."),
    ]
    engine = Engine(corpus, max_hops=3)
    state = engine.run("What is the policy procedure limit?")
    assert state.is_sufficient is True


def test_cycle_avoidance_structure():
    # Ensures the loop structure is deterministic and terminates at max_hops
    corpus = [Document(doc_id="1", text="trigger data")]
    engine = Engine(corpus, max_hops=3)
    state = engine.run("trigger")
    assert len(state.raw_memory) > 0
    assert state.hop_count == 3
    assert state.is_sufficient is False