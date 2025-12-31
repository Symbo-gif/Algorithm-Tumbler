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

import pytest
from algorithms.core.types import INT, FLOAT, BOOL, LIST
from algorithms.core.node import DSLPrimitive
from algorithms.core.program import Program
from algorithms.core.interpreter import Interpreter

def test_types():
    assert INT.name == "int"
    assert LIST(INT).element_type == INT

def test_dsl_primitives():
    # Test BinOpNode
    left = DSLPrimitive([], INT, [])
    right = DSLPrimitive([], INT, [])
    binop = DSLPrimitive([INT, INT], INT, [left, right])
    assert binop.input_types == [INT, INT]
    assert binop.output_type == INT

def test_program_validation():
    # Valid program
    node1 = DSLPrimitive([], INT, [])
    node2 = DSLPrimitive([INT], INT, [node1])
    prog = Program([node1, node2], {node1: [node2]})
    assert len(prog.nodes) == 2

    # Invalid: type mismatch
    node3 = DSLPrimitive([FLOAT], INT, [node1])  # INT != FLOAT
    with pytest.raises(TypeError):
        Program([node1, node3], {node1: [node3]})

def test_interpreter():
    interpreter = Interpreter()
    # Simple program
    node = DSLPrimitive([], INT, [])
    node.execute = lambda args: 42
    prog = Program([node], {})
    result = interpreter.execute(prog, {})
    assert result == 42
