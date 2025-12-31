from abc import ABC, abstractmethod
from typing import List
from ..core.node import Component

class DecompositionStrategy(ABC):
    """Base strategy for breaking code into reusable pieces."""

    @abstractmethod
    def decompose(self, code: str) -> List[Component]:
        """Return list of extracted components."""
        pass
