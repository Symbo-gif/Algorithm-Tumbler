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

from typing import List, Dict, Any, Callable, Optional
from .types import Type

class ASTNode:
    """Base AST node with type system."""

    def __init__(self, input_types: List[Type], output_type: Type, children: Optional[List['ASTNode']] = None):
        self.input_types = input_types
        self.output_type = output_type
        self.children = children or []
        self.semantic_spec: Optional[Callable] = None  # Optional logical spec

    def execute(self, args: List[Any]) -> Any:
        """Execute the node with given arguments."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(inputs={self.input_types}, output={self.output_type})"

class DSLPrimitive(ASTNode):
    """A first-class DSL constructor."""

    def __init__(self, input_types: List[Type], output_type: Type, children: Optional[List[ASTNode]] = None):
        super().__init__(input_types, output_type, children)

class Component:
    """Reusable component wrapper."""

    def __init__(self, pattern: ASTNode, parameter_schema: Dict[str, Type], cost: float,
                 spec: Optional[Any] = None, frequency: int = 0, success_rate: float = 0.0):
        self.pattern = pattern
        self.parameter_schema = parameter_schema
        self.cost = cost  # size/complexity estimate
        self.spec = spec  # test suite, correctness claim
        self.frequency = frequency
        self.success_rate = success_rate

    def instantiate(self) -> ASTNode:
        """Return a copy of the pattern for use."""
        # For now, just return the pattern; in full impl, handle parameters
        return self.pattern

    def __repr__(self):
        return f"Component(cost={self.cost}, freq={self.frequency})"
