from typing import Set, Tuple, List, Dict
import ast

class Node:
    """CFG node representing a statement or condition."""
    def __init__(self, stmt: ast.stmt):
        self.stmt = stmt
        self.id = id(stmt)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

class ControlFlowGraph:
    """Build CFG from AST, track data and control dependencies."""

    def __init__(self, ast_node: ast.AST):
        self.nodes: List[Node] = []
        self.edges: Dict[Node, List[Node]] = {}
        self.def_use: Dict[str, Set[Node]] = {}  # var -> nodes that define it
        self.use_def: Dict[str, Set[Node]] = {}  # var -> nodes that use it
        self._build_cfg(ast_node)

    def _build_cfg(self, ast_node: ast.AST):
        # Simplified CFG builder for function bodies
        if isinstance(ast_node, ast.FunctionDef):
            body = ast_node.body
            for i, stmt in enumerate(body):
                node = Node(stmt)
                self.nodes.append(node)
                if i > 0:
                    prev_node = self.nodes[i-1]
                    self.edges.setdefault(prev_node, []).append(node)
                # Build def-use
                self._analyze_stmt(node, stmt)

    def _analyze_stmt(self, node: Node, stmt: ast.stmt):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    self.def_use.setdefault(target.id, set()).add(node)
        elif isinstance(stmt, ast.Name):
            self.use_def.setdefault(stmt.id, set()).add(node)

    def backward_slice(self, criterion: Tuple[Node, Set[str]]) -> Set[Node]:
        """Program slicing: extract minimal code for criterion (node, variables).

        Returns all statements that affect the given variables at the given point.
        """
        node, vars_of_interest = criterion
        reachable: Set[Node] = set()
        stack = [node]

        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)

            # Add data dependencies: statements that define vars we use
            for var in vars_of_interest:
                for def_node in self.def_use.get(var, set()):
                    if def_node != current:  # Avoid self
                        stack.append(def_node)

            # Add control dependencies: conditions that control this node
            for ctrl_dep in self._find_control_deps(current):
                stack.append(ctrl_dep)

        return reachable

    def forward_slice(self, criterion: Tuple[Node, Set[str]]) -> Set[Node]:
        """Forward slice: all code affected by given variables at given point."""
        node, vars_of_interest = criterion
        reachable: Set[Node] = set()
        stack = [node]

        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)

            # Add data dependencies: statements that use vars we define
            for var in vars_of_interest:
                for use_node in self.use_def.get(var, set()):
                    if use_node != current:
                        stack.append(use_node)

        return reachable

    def _find_control_deps(self, node: Node) -> List[Node]:
        """Find control dependencies (simplified)."""
        # In full impl, use post-dominator analysis
        return []

    def extract_independent_components(self) -> List[Set[Node]]:
        """Decomposition slicing: partition code into independent components."""
        components = []
        visited = set()
        for node in self.nodes:
            if node not in visited:
                component = self._dfs_component(node, visited)
                components.append(component)
        return components

    def _dfs_component(self, start: Node, visited: Set[Node]) -> Set[Node]:
        component = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in self.edges.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        return component
