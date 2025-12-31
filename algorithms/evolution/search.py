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

from typing import List, Dict, Any
import random
from ..core.program import Program
from .mutations import subtree_mutation, crossover, guided_mutation
from .fitness import fitness

class EvolutionarySearch:
    def __init__(self, library: 'ComponentLibrary', seed: int):
        self.library = library
        self.prng = random.Random(seed)
        self.population: List[Program] = []
        self.hall_of_fame = []

    def initialize(self, size: int, max_depth: int = 5):
        """Random initialization respecting type constraints."""
        for _ in range(size):
            prog = self._random_program(max_depth)
            self.population.append(prog)

    def _random_program(self, max_depth: int) -> Program:
        # Placeholder: create a simple program
        from ..core.node import DSLPrimitive
        from ..core.types import INT
        node = DSLPrimitive([], INT, [])
        return Program([node], {})

    def evolve(self, generations: int, fitness_fn, test_cases: List[Tuple[Any, Any]], mode="random"):
        """
        mode="random": random mutations
        mode="guided": mutation biased by objective_spec
        """
        for gen in range(generations):
            # Evaluate
            fitnesses = [fitness_fn(p, test_cases) for p in self.population]

            # Select (tournament)
            parents = self._tournament_selection(self.population, fitnesses, k=5)

            # Vary
            new_pop = []
            for parent in parents:
                if mode == "guided":
                    child = guided_mutation(parent, "sort", self.prng, self.library)  # Placeholder spec
                else:
                    if self.prng.random() < 0.5:
                        child = subtree_mutation(parent, self.prng, self.library)
                    else:
                        child = crossover(parent, self.prng.choice(parents), self.prng)
                new_pop.append(child)

            self.population = new_pop

            # Track best
            best = max(self.population, key=lambda p: fitness_fn(p, test_cases))
            self.hall_of_fame.append(best)

            if gen % 10 == 0:
                print(f"Gen {gen}: Best fitness = {fitness_fn(best, test_cases):.2f}")

    def _tournament_selection(self, population: List[Program], fitnesses: List[float], k: int) -> List[Program]:
        selected = []
        for _ in range(len(population)):
            candidates = self.prng.sample(list(zip(population, fitnesses)), k)
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected
