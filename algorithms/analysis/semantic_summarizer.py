from typing import List, Optional

class SemanticSummarizer:
    """Use LLM to summarize code purpose when static analysis is insufficient."""

    def summarize(self, code: str, context: Optional[str] = None) -> str:
        """Generate English description of what code does."""
        # Placeholder: in real impl, call LLM
        return f"Code summary: {code[:50]}..."

    def extract_invariants(self, code: str) -> List[str]:
        """Identify loop invariants, pre/post-conditions."""
        # Placeholder
        return ["Invariant: x >= 0"]
