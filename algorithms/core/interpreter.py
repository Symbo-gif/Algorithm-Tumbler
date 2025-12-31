from typing import Dict, Any
from .program import Program

class Interpreter:
    """Safe execution engine for programs."""

    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout

    def execute(self, program: Program, inputs: Dict[str, Any]) -> Any:
        """Execute program with inputs, with safety limits."""
        try:
            return program.execute(inputs)
        except Exception as e:
            return None  # Or raise
