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

"""Test the tumbling/decomposition functionality."""

from algorithms.decomposer.function_extractor import FunctionLevelDecomposer


def test_function_extractor_basic():
    """Test that function extractor can parse and extract functions."""
    code = """
def add(x, y):
    '''Add two numbers'''
    return x + y

def multiply(a, b):
    '''Multiply two numbers'''
    return a * b
"""
    decomposer = FunctionLevelDecomposer()
    components = decomposer.decompose(code)
    
    assert len(components) == 2
    assert components[0].pattern.name == "add"
    assert components[1].pattern.name == "multiply"
    assert "Add two numbers" in components[0].spec
    assert "Multiply two numbers" in components[1].spec


def test_function_extractor_with_params():
    """Test that function extractor handles parameters correctly."""
    code = """
def process(data, flag, count):
    '''Process data with flag and count'''
    return data
"""
    decomposer = FunctionLevelDecomposer()
    components = decomposer.decompose(code)
    
    assert len(components) == 1
    comp = components[0]
    assert comp.pattern.name == "process"
    assert len(comp.parameter_schema) == 3
    assert "data" in comp.parameter_schema
    assert "flag" in comp.parameter_schema
    assert "count" in comp.parameter_schema


def test_function_extractor_syntax_error():
    """Test that function extractor handles syntax errors gracefully."""
    code = "def broken(x y):"  # Invalid syntax
    decomposer = FunctionLevelDecomposer()
    components = decomposer.decompose(code)
    
    assert len(components) == 0  # Should return empty list on syntax error


def test_function_extractor_no_functions():
    """Test that function extractor handles code with no functions."""
    code = """
x = 10
y = 20
z = x + y
"""
    decomposer = FunctionLevelDecomposer()
    components = decomposer.decompose(code)
    
    assert len(components) == 0
