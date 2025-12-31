import ast
from typing import List
from ..core.node import Component, DSLPrimitive
from ..core.types import INT, LIST
from .strategies import DecompositionStrategy

class LoopExtractor(DecompositionStrategy):
    """Extract loop bodies and patterns as components."""

    def decompose(self, code: str) -> List[Component]:
        tree = ast.parse(code)
        components = []

        for loop in ast.walk(tree):
            if isinstance(loop, (ast.For, ast.While)):
                # Extract loop body as standalone component
                body = self._extract_body(loop)
                loop_comp = Component(
                    pattern=DSLPrimitive([], INT, []),  # Placeholder
                    name=f"loop_{id(loop)}",
                    spec={'pattern': 'loop_body'}
                )
                components.append(loop_comp)

                # Extract specific idioms: foreach, map, filter, reduce
                if self._looks_like_foreach(loop):
                    idiom = self._extract_foreach_pattern(loop)
                    components.append(idiom)

                if self._looks_like_map(loop):
                    idiom = self._extract_map_pattern(loop)
                    components.append(idiom)

        return components

    def _extract_body(self, loop: ast.stmt) -> ast.stmt:
        if isinstance(loop, ast.For):
            return loop.body
        elif isinstance(loop, ast.While):
            return loop.body

    def _looks_like_foreach(self, loop: ast.stmt) -> bool:
        # Simplified check
        return isinstance(loop, ast.For)

    def _looks_like_map(self, loop: ast.stmt) -> bool:
        # Simplified
        return False

    def _extract_foreach_pattern(self, loop: ast.For) -> Component:
        return Component(
            pattern=DSLPrimitive([], LIST(INT), []),
            name="foreach",
            spec={'pattern': 'foreach'}
        )

    def _extract_map_pattern(self, loop: ast.For) -> Component:
        return Component(
            pattern=DSLPrimitive([], LIST(INT), []),
            name="map",
            spec={'pattern': 'map'}
        )
