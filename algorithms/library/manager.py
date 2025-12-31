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

from typing import Dict, List, Optional, Iterable
from ..core.node import Component
from ..core.types import Type
from .seeds import seed_components, seed_builders


def _type_key(t: Optional[Type]) -> str:
    """Stable representation to compare compatible types."""
    return repr(t)


class ComponentLibrary:
    """Lightweight in-memory library for reusable components."""

    def __init__(self):
        self.components: Dict[str, Component] = {}

    @classmethod
    def seed_default(cls) -> "ComponentLibrary":
        """Build a library populated with a handful of seed algorithms."""
        lib = cls()
        for cid, comp in seed_components().items():
            lib.add(comp, cid)
        return lib

    def add(self, component: Component, component_id: Optional[str] = None) -> str:
        cid = component_id or f"comp_{len(self.components)}"
        self.components[cid] = component
        return cid

    def query_by_type(self, output_type: Type) -> List[Component]:
        target = _type_key(output_type)
        return [
            comp
            for comp in self.components.values()
            if _type_key(comp.pattern.output_type) == target
        ]

    def query_by_spec(self, spec: str) -> List[Component]:
        needle = spec.lower()
        return [
            comp
            for comp in self.components.values()
            if comp.spec and needle in str(comp.spec).lower()
        ]

    def bulk_add(self, components: Iterable[Component]):
        for comp in components:
            self.add(comp)

    def save(self, path: str):
        from ..io.serialization import library_to_dict
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(library_to_dict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ComponentLibrary":
        from ..io.serialization import library_from_dict
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return library_from_dict(data)

    def get(self, component_id: str) -> Component:
        return self.components[component_id]

    def run(self, component_id: str, *args):
        """Execute a stored component directly with Python values."""
        component = self.get(component_id)
        node = component.instantiate()
        return node.execute(list(args))

    @property
    def seed_ids(self) -> List[str]:
        return list(seed_builders().keys())
