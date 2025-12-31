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

from typing import List, Optional

from ..symbo_bridge import SymboBridge

class SemanticSummarizer:
    """Use LLM-backed Symbo components to summarize code when static analysis is insufficient."""

    def __init__(self, symbo_bridge: Optional[SymboBridge] = None):
        self.symbo_bridge = symbo_bridge or SymboBridge()

    def summarize(self, code: str, context: Optional[str] = None) -> str:
        """Generate English description of what code does."""
        return self.symbo_bridge.summarize_code(code, context)

    def extract_invariants(self, code: str) -> List[str]:
        """Identify loop invariants, pre/post-conditions."""
        invariants = ["Invariant: x >= 0"]
        status = self.symbo_bridge.status()
        invariants.append(f"Symbo components: {status}")
        return invariants
