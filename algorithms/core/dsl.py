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

from .node import DSLPrimitive
from .types import INT, FLOAT, BOOL, LIST, STRING

# Control flow
class IfNode(DSLPrimitive):
    def __init__(self, cond: DSLPrimitive, then_branch: DSLPrimitive, else_branch: DSLPrimitive):
        super().__init__([BOOL, then_branch.output_type, else_branch.output_type], then_branch.output_type, [cond, then_branch, else_branch])

    def execute(self, args):
        cond, then_val, else_val = args
        return then_val if cond else else_val

class ForNode(DSLPrimitive):
    def __init__(self, iterable: DSLPrimitive, body: DSLPrimitive):
        # Simplified: body takes current item
        super().__init__([LIST(iterable.output_type.element_type), body.input_types[0]], LIST(body.output_type), [iterable, body])

    def execute(self, args):
        iterable, body = args
        result = []
        for item in iterable:
            result.append(body.execute([item]))
        return result

class WhileNode(DSLPrimitive):
    def __init__(self, cond: DSLPrimitive, body: DSLPrimitive):
        super().__init__([BOOL, body.output_type], body.output_type, [cond, body])

    def execute(self, args):
        cond, body = args
        while cond.execute([]):
            body.execute([])
        return None  # Simplified

# Data ops
class IndexNode(DSLPrimitive):
    def __init__(self, container: DSLPrimitive, index: DSLPrimitive):
        super().__init__([LIST(container.output_type.element_type), INT], container.output_type.element_type, [container, index])

    def execute(self, args):
        container, index = args
        return container[index]

class SliceNode(DSLPrimitive):
    def __init__(self, container: DSLPrimitive, start: DSLPrimitive, end: DSLPrimitive):
        super().__init__([LIST(container.output_type.element_type), INT, INT], LIST(container.output_type.element_type), [container, start, end])

    def execute(self, args):
        container, start, end = args
        return container[start:end]

class MapNode(DSLPrimitive):
    def __init__(self, fn: DSLPrimitive, iterable: DSLPrimitive):
        super().__init__([fn.input_types[0], LIST(iterable.output_type.element_type)], LIST(fn.output_type), [fn, iterable])

    def execute(self, args):
        fn, iterable = args
        return [fn.execute([item]) for item in iterable]

class ReduceNode(DSLPrimitive):
    def __init__(self, fn: DSLPrimitive, iterable: DSLPrimitive, init: DSLPrimitive):
        super().__init__([fn.input_types[0], fn.input_types[1], LIST(iterable.output_type.element_type)], fn.output_type, [fn, iterable, init])

    def execute(self, args):
        fn, iterable, init = args
        result = init
        for item in iterable:
            result = fn.execute([result, item])
        return result

# Arithmetic
class BinOpNode(DSLPrimitive):
    def __init__(self, left: DSLPrimitive, op: str, right: DSLPrimitive):
        if op in ['+', '-', '*', '/', '//', '%']:
            output_type = INT if left.output_type == INT and right.output_type == INT else FLOAT
        elif op in ['&', '|', '^']:
            output_type = INT
        else:
            raise ValueError(f"Unsupported op: {op}")
        super().__init__([left.output_type, right.output_type], output_type, [left, right])
        self.op = op

    def execute(self, args):
        left, right = args
        if self.op == '+':
            return left + right
        elif self.op == '-':
            return left - right
        elif self.op == '*':
            return left * right
        elif self.op == '/':
            return left / right
        elif self.op == '//':
            return left // right
        elif self.op == '%':
            return left % right
        elif self.op == '&':
            return left & right
        elif self.op == '|':
            return left | right
        elif self.op == '^':
            return left ^ right

class UnaryOpNode(DSLPrimitive):
    def __init__(self, op: str, operand: DSLPrimitive):
        if op == '-':
            output_type = operand.output_type
        elif op == '~':
            output_type = INT
        else:
            raise ValueError(f"Unsupported unary op: {op}")
        super().__init__([operand.output_type], output_type, [operand])
        self.op = op

    def execute(self, args):
        operand = args[0]
        if self.op == '-':
            return -operand
        elif self.op == '~':
            return ~operand

# I/O
class LoadNode(DSLPrimitive):
    def __init__(self, input_type: type):
        super().__init__([], input_type, [])

    def execute(self, args):
        # In real impl, this would load from input dict
        raise NotImplementedError

class StoreNode(DSLPrimitive):
    def __init__(self, value: DSLPrimitive):
        super().__init__([value.output_type], value.output_type, [value])

    def execute(self, args):
        # In real impl, this would store to output
        return args[0]
