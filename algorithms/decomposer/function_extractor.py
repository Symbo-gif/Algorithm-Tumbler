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
from typing import List, Callable, Any
from ..core.node import Component, DSLPrimitive
from ..core.types import Type, INT, LIST, BOOL, STRING
from .strategies import DecompositionStrategy


class FunctionPrimitive(DSLPrimitive):
    """DSL primitive that wraps a Python function."""

    def __init__(self, name: str, input_types: List[Type], output_type: Type, fn: Callable):
        super().__init__(input_types, output_type, [])
        self._fn = fn
        self.name = name

    def execute(self, args: List[Any]) -> Any:
        return self._fn(*args)

    def __repr__(self):
        return f"FunctionPrimitive({self.name})"


class FunctionLevelDecomposer(DecompositionStrategy):
    """Extract functions from Python code as reusable components."""

    def decompose(self, code: str) -> List[Component]:
        """Parse Python code and extract all function definitions as components."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        components = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                comp = self._extract_function(node, code)
                if comp:
                    components.append(comp)

        return components

    def _extract_function(self, func_node: ast.FunctionDef, source_code: str) -> Component:
        """Convert an AST function node into a Component."""
        func_name = func_node.name
        
        # Extract docstring for spec
        spec = ast.get_docstring(func_node) or f"function {func_name}"
        spec = spec.strip().split('\n')[0][:100]  # First line, max 100 chars
        
        # Infer types (simplified - use generic types)
        num_params = len(func_node.args.args)
        input_types = [INT] * num_params  # Default to INT for simplicity
        output_type = INT  # Default return type
        
        # Create a placeholder function (actual execution would require more complex setup)
        def placeholder_fn(*args):
            raise NotImplementedError(f"Extracted function {func_name} cannot be executed directly")
        
        primitive = FunctionPrimitive(
            name=func_name,
            input_types=input_types,
            output_type=output_type,
            fn=placeholder_fn
        )
        
        # Build parameter schema
        param_names = [arg.arg for arg in func_node.args.args]
        parameter_schema = {name: typ for name, typ in zip(param_names, input_types)}
        
        component = Component(
            pattern=primitive,
            parameter_schema=parameter_schema,
            cost=2.0 + num_params * 0.5,  # Cost based on complexity
            spec=spec,
            frequency=1,  # New component
            success_rate=0.5,  # Unknown success rate
        )
        
        return component
