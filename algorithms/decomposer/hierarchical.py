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

from typing import List
from ..core.node import Component
from .strategies import DecompositionStrategy
from .function_extractor import FunctionLevelDecomposer
from .loop_extractor import LoopExtractor
from .dataflow_extractor import DataFlowExtractor

class HierarchicalDecomposer(DecompositionStrategy):
    """Multi-level: functions -> loops -> data flows -> idioms."""

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def decompose(self, code: str) -> List[Component]:
        """Recursive decomposition at multiple granularities."""
        components = []

        level_0 = FunctionLevelDecomposer().decompose(code)
        components.extend(level_0)

        for func_comp in level_0:
            # Simulate extracting from func_comp.pattern
            level_1 = LoopExtractor().decompose(code)  # Placeholder
            for loop_comp in level_1:
                loop_comp.parent = func_comp  # Track hierarchy
                components.append(loop_comp)

            level_2 = DataFlowExtractor().decompose(code)  # Placeholder
            for df_comp in level_2:
                df_comp.parent = func_comp
                components.append(df_comp)

        return components
