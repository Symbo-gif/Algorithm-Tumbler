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
from typing import Dict
from ..core.node import Component
from ..library.manager import ComponentLibrary

def component_to_dict(comp: Component) -> Dict:
    return {
        'pattern': str(comp.pattern),  # Placeholder
        'parameter_schema': comp.parameter_schema,
        'cost': comp.cost,
        'spec': comp.spec,
        'frequency': comp.frequency,
        'success_rate': comp.success_rate
    }

def component_from_dict(data: Dict) -> Component:
    return Component(
        pattern=None,  # Placeholder
        parameter_schema=data['parameter_schema'],
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
        comp = component_from_dict(comp_data)
        lib.add(comp)
    return lib
