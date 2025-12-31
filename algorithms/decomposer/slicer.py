import ast
from typing import List, Set
from ..core.node import Component, DSLPrimitive
from ..core.types import INT, LIST
from ..analysis.program_slicing import ControlFlowGraph, Node
from .strategies import DecompositionStrategy

class SlicingDecomposer(DecompositionStrategy):
    """Use program slicing to extract minimal semantic units."""

    def decompose(self, code: str) -> List[Component]:
        """For each 'interesting' variable/output, extract its slice."""
        tree = ast.parse(code)
        cfg = ControlFlowGraph(tree)
        components = []

        # Find all output points (returns, prints, assignments to globals)
        for output_point in cfg.nodes:
            if self._is_output_point(output_point.stmt):
                for var in self._extract_vars(output_point.stmt):
                    # Backward slice
                    slice_nodes = cfg.backward_slice((output_point, {var}))
                    comp = Component(
                        pattern=self._nodes_to_ast(slice_nodes),
                        name=f"slice_{var}_{id(output_point)}",
                        computes=var
                    )
                    components.append(comp)

        return components

    def _is_output_point(self, stmt: ast.stmt) -> bool:
        return isinstance(stmt, (ast.Return, ast.Expr))  # Simplified

    def _extract_vars(self, stmt: ast.stmt) -> Set[str]:
        vars = set()
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                vars.add(node.id)
        return vars

    def _nodes_to_ast(self, nodes: Set[Node]) -> DSLPrimitive:
        # Simplified: return a dummy node
        return DSLPrimitive([], INT, [])
