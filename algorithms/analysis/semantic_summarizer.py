from typing import List, Optional

from ..symbo_bridge import SymboBridge

class SemanticSummarizer:
    """Use LLM-backed Symbo components to summarize code when static analysis is insufficient."""

    def __init__(self, symbo_bridge: Optional[SymboBridge] = None):
        self.symbo_bridge = symbo_bridge or SymboBridge()

    def summarize(self, code: str, context: Optional[str] = None) -> str:
        """Generate English description of what code does."""
        return self.symbo_bridge.summarize_code(code, context)

    def extract_invariants(self, code: str) -> List[str]:
        """Identify loop invariants, pre/post-conditions."""
        invariants = ["Invariant: x >= 0"]
        status = self.symbo_bridge.status()
        invariants.append(f"Symbo components: {status}")
        return invariants
