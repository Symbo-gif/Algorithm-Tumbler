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

from typing import List, Any, Dict
import copy
from .node import ASTNode
from .types import Type

class Program:
    """Program = acyclic directed graph of nodes."""

    def __init__(self, nodes: List[ASTNode], edges: Dict[ASTNode, List[ASTNode]]):
        self.nodes = nodes
        self.edges = edges  # adjacency list
        self._validate()

    def _validate(self):
        """Check for cycles and type consistency."""
        # Simple cycle check (DFS)
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self.edges.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if has_cycle(node):
                    raise ValueError("Program contains cycles")

        # Type check: ensure inputs match outputs
        for node in self.nodes:
            for i, child in enumerate(node.children):
                if child.output_type != node.input_types[i]:
                    raise TypeError(f"Type mismatch in {node}: expected {node.input_types[i]}, got {child.output_type}")

    def execute(self, inputs: Dict[str, Any]) -> Any:
        """Execute the program with given inputs."""
        # Simple topological execution
        memo = {}
        def exec_node(node):
            if node in memo:
                return memo[node]
            args = [exec_node(child) for child in node.children]
            result = node.execute(args)
            memo[node] = result
            return result

        # Assume the last node is the output
        if self.nodes:
            return exec_node(self.nodes[-1])
        return None

    def ast_size(self) -> int:
        """Return the size of the AST."""
        return sum(1 for node in self.nodes for _ in node.children) + len(self.nodes)

    def copy(self) -> 'Program':
        """Return a deep copy of the program."""
        return Program(copy.deepcopy(self.nodes), copy.deepcopy(self.edges))

    def extract_subtree(self, target: ASTNode) -> ASTNode:
        """Return a copy of the subtree rooted at target."""
        return copy.deepcopy(target)

    def substitute_subtree(self, target: ASTNode, replacement: ASTNode) -> 'Program':
        """Create a new program with one subtree replaced."""
        new_nodes = []
        for node in self.nodes:
            new_nodes.append(replacement if node is target else node)

        new_edges: Dict[ASTNode, List[ASTNode]] = {}
        for parent, children in self.edges.items():
            new_parent = replacement if parent is target else parent
            new_children = [replacement if child is target else child for child in children]
            new_edges[new_parent] = new_children

        # Update child references inside nodes
        for node in new_nodes:
            node.children = [replacement if child is target else child for child in node.children]

        return Program(new_nodes, new_edges)

    def __repr__(self):
        return f"Program(nodes={len(self.nodes)})"
