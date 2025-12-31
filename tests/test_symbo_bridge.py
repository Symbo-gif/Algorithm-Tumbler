from algorithms.symbo_bridge import SymboBridge
from algorithms.analysis.semantic_summarizer import SemanticSummarizer


def test_symbo_bridge_simplify_and_status():
    bridge = SymboBridge(enable_llm=False)
    status = bridge.status()
    assert "llm_available" in status
    result = bridge.simplify_expression("1 + 2 * 3")
    assert str(result["result"]).startswith("7")


def test_symbo_bridge_story_reasoning():
    bridge = SymboBridge(enable_llm=False)
    analysis = bridge.reason_about_story("Alice has 3 apples and gets 2 more.")
    assert "entities" in analysis
    assert len(analysis["entities"]) >= 2


def test_semantic_summarizer_with_symbo_bridge():
    bridge = SymboBridge(enable_llm=False)
    summarizer = SemanticSummarizer(symbo_bridge=bridge)
    summary = summarizer.summarize("def foo():\n    return 42")
    assert summary.lower().startswith("symbo")
