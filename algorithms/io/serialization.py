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

import json
from typing import Dict, List
from ..core.node import Component, DSLPrimitive
from ..core.types import PrimitiveType, ContainerType, Type, INT, FLOAT, BOOL, STRING, LIST, DICT
from ..library.manager import ComponentLibrary
from ..library.seeds import seed_builders


def _type_to_string(t: Type) -> str:
    if isinstance(t, PrimitiveType):
        return t.name
    if isinstance(t, ContainerType):
        elem = t.element_type
        if isinstance(elem, tuple) and len(elem) == 2:
            return f"dict[{_type_to_string(elem[0])}:{_type_to_string(elem[1])}]"
        return f"list[{_type_to_string(elem)}]"
    return str(t)


def _type_from_string(s: str) -> Type:
    if s == "int":
        return INT
    if s == "float":
        return FLOAT
    if s == "bool":
        return BOOL
    if s == "str":
        return STRING
    if s.startswith("list[") and s.endswith("]"):
        inner = s[5:-1]
        return LIST(_type_from_string(inner))
    if s.startswith("dict[") and s.endswith("]") and ":" in s:
        inner = s[5:-1]
        key_str, val_str = inner.split(":", 1)
        return DICT(_type_from_string(key_str), _type_from_string(val_str))
    return STRING

def component_to_dict(comp: Component) -> Dict:
    return {
        'pattern': str(comp.pattern),
        'parameter_schema': {k: _type_to_string(v) for k, v in comp.parameter_schema.items()},
        'cost': comp.cost,
        'spec': comp.spec,
        'frequency': comp.frequency,
        'success_rate': comp.success_rate,
        'output_type': _type_to_string(comp.pattern.output_type) if comp.pattern else None,
        'input_types': [
            _type_to_string(t) for t in getattr(comp.pattern, "input_types", []) or []
        ],
    }

def component_from_dict(data: Dict, component_id: str = "") -> Component:
    # Prefer rebuilding from known seeds when possible for faithful reconstruction
    builders = seed_builders()
    if component_id in builders:
        return builders[component_id]()

    output_type = _type_from_string(data.get('output_type')) if data.get('output_type') else STRING
    input_types: List[Type] = [
        _type_from_string(t) for t in data.get('input_types', [])
    ]
    pattern = DSLPrimitive(input_types, output_type, [])
    pattern.execute = lambda args: None
    return Component(
        pattern=pattern,
        parameter_schema={k: _type_from_string(v) for k, v in data['parameter_schema'].items()},
        cost=data['cost'],
        spec=data['spec'],
        frequency=data['frequency'],
        success_rate=data['success_rate']
    )

def library_to_dict(lib: ComponentLibrary) -> Dict:
    return {
        'components': {cid: component_to_dict(comp) for cid, comp in lib.components.items()}
    }

def library_from_dict(data: Dict) -> ComponentLibrary:
    lib = ComponentLibrary()
    for cid, comp_data in data['components'].items():
        comp = component_from_dict(comp_data, cid)
        lib.add(comp, cid)
    return lib
