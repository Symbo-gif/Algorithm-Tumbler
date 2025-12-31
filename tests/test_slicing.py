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
