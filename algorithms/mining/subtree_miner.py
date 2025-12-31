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

from typing import List, Dict
from ..core.node import ASTNode

class ProbabilisticTreeSubstitutionGrammar:
    """Probabilistic Tree Substitution Grammar for mining idioms."""

    def __init__(self):
        self.rules: Dict[str, List[ASTNode]] = {}  # nonterminal -> possible subtrees
        self.probabilities: Dict[str, Dict[str, float]] = {}

    def learn(self, subtrees: List[ASTNode]):
        """Learn grammar from subtrees."""
        # Simplified: group by structure
        for subtree in subtrees:
            key = self._subtree_key(subtree)
            if key not in self.rules:
                self.rules[key] = []
            self.rules[key].append(subtree)

    def _subtree_key(self, node: ASTNode) -> str:
        # Simplified key
        return str(type(node))

    def extract_idioms(self) -> List[ASTNode]:
        """Return high-probability fragments = idioms."""
        idioms = []
        for key, trees in self.rules.items():
            if len(trees) > 1:  # Frequent
                idioms.append(trees[0])  # Representative
        return idioms
