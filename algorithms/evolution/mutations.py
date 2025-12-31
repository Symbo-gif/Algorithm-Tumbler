from typing import List
import random
from ..core.node import ASTNode, Component
from ..core.program import Program
from ..core.types import Type

def subtree_mutation(program: Program, prng: random.Random, library: 'ComponentLibrary') -> Program:
    """Pick random subtree, replace with compatible component or randomize."""
    if not program.nodes:
        return program
    target_node = prng.choice(program.nodes)
    output_type = target_node.output_type

    # Option 1: Replace with library component of matching type
    candidates = library.query_by_type(output_type) if hasattr(library, 'query_by_type') else []
    if candidates and prng.random() < 0.7:
        component = prng.choice(candidates)
        return program.substitute_subtree(target_node, component.instantiate())

    # Option 2: Random mutation of node parameters (simplified)
    return program  # Placeholder

def crossover(p1: Program, p2: Program, prng: random.Random) -> Program:
    """Swap compatible subtrees between p1 and p2."""
    if not p1.nodes or not p2.nodes:
        return p1
    node1 = prng.choice(p1.nodes)
    # Find all nodes in p2 with same output type
    compatible = [n for n in p2.nodes if n.output_type == node1.output_type]
    if not compatible:
        return p1.copy()  # No crossover possible
    node2 = prng.choice(compatible)
    return p1.substitute_subtree(node1, p2.extract_subtree(node2))

def guided_mutation(program: Program, objective_spec: str, prng: random.Random, library: 'ComponentLibrary') -> Program:
    """Mutation biased toward satisfying objective_spec."""
    # Try to insert components known to help with objective_spec
    for _ in range(5):
        if not program.nodes:
            continue
        target = prng.choice(program.nodes)
        suitable = library.query_by_spec(objective_spec) if hasattr(library, 'query_by_spec') else []
        if suitable:
            comp = prng.choice(suitable)
            program = program.substitute_subtree(target, comp.instantiate())
    return program
