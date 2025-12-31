# Copyright 2025 Damien Davison, Michael Maillet, and Sacha Davison, Recursive AI Devs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
from typing import List
from ..core.node import Component, DSLPrimitive
from ..core.types import INT
from .strategies import DecompositionStrategy


class DataFlowExtractor(DecompositionStrategy):
    """Extract data flow patterns from code."""

    def decompose(self, code: str) -> List[Component]:
        """Extract data flow components from Python code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        components = []
        
        # Look for common data flow patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Extract assignment patterns
                comp = self._extract_assignment(node)
                if comp:
                    components.append(comp)

        return components

    def _extract_assignment(self, assign_node: ast.Assign) -> Component:
        """Extract a simple assignment as a data flow component."""
        # Simplified: create a placeholder component
        primitive = DSLPrimitive([], INT, [])
        
        component = Component(
            pattern=primitive,
            parameter_schema={},
            cost=1.0,
            spec="data flow assignment",
            frequency=1,
            success_rate=0.5,
        )
        
        return component
