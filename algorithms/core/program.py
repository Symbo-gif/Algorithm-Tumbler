from typing import List, Any, Dict
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

    def __repr__(self):
        return f"Program(nodes={len(self.nodes)})"
