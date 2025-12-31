import ast
from algorithms.analysis.program_slicing import ControlFlowGraph

def test_cfg_build():
    code = """
def foo():
    x = 1
    y = x + 1
    return y
"""
    tree = ast.parse(code)
    cfg = ControlFlowGraph(tree)
    assert len(cfg.nodes) > 0
