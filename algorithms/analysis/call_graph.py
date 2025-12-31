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
