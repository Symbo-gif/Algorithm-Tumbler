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

from typing import Any, List, Tuple


def fitness(program: 'Program', test_cases: List[Tuple[Any, Any]]) -> float:
    """
    Evaluate a program's fitness based on test cases.
    
    Args:
        program: The program to evaluate
        test_cases: List of (input, expected_output) tuples
        
    Returns:
        Fitness score (higher is better)
    """
    score = 0.0
    
    for inputs, expected in test_cases:
        try:
            result = program.execute(inputs if isinstance(inputs, dict) else {'input': inputs})
            if result == expected:
                score += 1.0
        except Exception:
            # Execution failed
            score -= 0.1
    
    # Penalize complexity
    try:
        complexity_penalty = program.ast_size() * 0.01
        score -= complexity_penalty
    except Exception:
        pass
    
    return score
