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

from typing import Callable, Dict
from ..core.node import DSLPrimitive, Component
from ..core.types import INT, LIST, BOOL


class FunctionPrimitive(DSLPrimitive):
    """DSL primitive that delegates execution to a Python function."""

    def __init__(self, name: str, input_types, output_type, fn: Callable):
        super().__init__(input_types, output_type, [])
        self._fn = fn
        self.name = name

    def execute(self, args):
        return self._fn(*args)

    def __repr__(self):
        return f"FunctionPrimitive({self.name})"


_SEED_BUILDERS: Dict[str, Callable[[], Component]] = {}


def register_seed(name: str):
    def decorator(fn: Callable[[], Component]):
        _SEED_BUILDERS[name] = fn
        return fn

    return decorator


@register_seed("sort_int_list")
def build_sort_int_list() -> Component:
    node = FunctionPrimitive(
        "sort_int_list",
        [LIST(INT)],
        LIST(INT),
        lambda items: sorted(items),
    )
    return Component(
        pattern=node,
        parameter_schema={"items": LIST(INT)},
        cost=3.0,
        spec="sort integers ascending",
        frequency=25,
        success_rate=0.95,
    )


@register_seed("binary_search")
def build_binary_search() -> Component:
    def _binary_search(sorted_list, target):
        lo, hi = 0, len(sorted_list) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_list[mid] == target:
                return mid
            if sorted_list[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    node = FunctionPrimitive(
        "binary_search",
        [LIST(INT), INT],
        INT,
        _binary_search,
    )
    return Component(
        pattern=node,
        parameter_schema={"haystack": LIST(INT), "needle": INT},
        cost=4.0,
        spec="search integer in sorted list",
        frequency=18,
        success_rate=0.9,
    )


@register_seed("is_sorted")
def build_is_sorted() -> Component:
    def _is_sorted(items):
        return all(items[i] <= items[i + 1] for i in range(len(items) - 1))

    node = FunctionPrimitive(
        "is_sorted",
        [LIST(INT)],
        BOOL,
        _is_sorted,
    )
    return Component(
        pattern=node,
        parameter_schema={"items": LIST(INT)},
        cost=2.0,
        spec="check if list sorted",
        frequency=30,
        success_rate=0.97,
    )


@register_seed("factorial")
def build_factorial() -> Component:
    def _factorial(n: int) -> int:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    node = FunctionPrimitive(
        "factorial",
        [INT],
        INT,
        _factorial,
    )
    return Component(
        pattern=node,
        parameter_schema={"n": INT},
        cost=3.5,
        spec="math factorial",
        frequency=20,
        success_rate=0.92,
    )


def seed_components() -> Dict[str, Component]:
    """Instantiate all registered seed components."""
    return {name: builder() for name, builder in _SEED_BUILDERS.items()}


def seed_builders() -> Dict[str, Callable[[], Component]]:
    return dict(_SEED_BUILDERS)
