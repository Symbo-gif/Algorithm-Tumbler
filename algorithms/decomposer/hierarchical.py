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
