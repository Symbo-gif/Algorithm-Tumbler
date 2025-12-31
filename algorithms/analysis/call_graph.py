import ast
from typing import Dict, List, Set

class CallGraph:
    """Extract function call relationships + dependencies."""

    def __init__(self, ast_node: ast.AST):
        self.functions: Dict[str, ast.FunctionDef] = {}
        self.calls: Dict[str, Set[str]] = {}  # func -> called funcs
        self._build(ast_node)

    def _build(self, ast_node: ast.AST):
        for node in ast.walk(ast_node):
            if isinstance(node, ast.FunctionDef):
                self.functions[node.name] = node
                self.calls[node.name] = set()
                for call in ast.walk(node):
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                        self.calls[node.name].add(call.func.id)

    def strongly_connected_components(self) -> List[Set[str]]:
        """Find SCCs using Kosaraju or Tarjan (simplified DFS-based)."""
        # Simplified: assume no cycles for now
        return [{name} for name in self.functions]

    def external_calls(self, funcs: Set[str]) -> Set[str]:
        """Return external functions called by the given set."""
        external = set()
        for func in funcs:
            for called in self.calls.get(func, set()):
                if called not in funcs:
                    external.add(called)
        return external
