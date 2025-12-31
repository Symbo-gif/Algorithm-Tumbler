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

"""
ReasoningTensor: Modified NanoTensor for Word Problem Reasoning
================================================================

Specialized symbolic tensor engine for word problem reasoning with:
- Entity-Time State Matrix: Track entity values over narrative time
- Constraint Accumulator: Collect equations from text for solving
- Semantic Knowledge Base: Store facts about entities and relationships
- Sign Inference Engine: Determine operation polarity (add vs subtract)

This is a focused derivative of NanoTensor optimized for the GSM8K
benchmark error categories:
- Sign errors (10.9%)
- Variable tracking across sentences
- Comparative composition ("twice as many as X who has 4 times Y")
- Multi-step operation chaining

NO SYMPY - Uses native symbolic engine for all operations.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Native symbolic imports - NO SYMPY
from symbo_agentic_reasoners.core.native_symbolic import (
    Symbol, sympify, simplify, parse_expr,
    symbols as create_symbols, Rational, Integer
)
from symbo_agentic_reasoners.core.calculus import solve_polynomial as native_solve


class OperationType(Enum):
    """Types of arithmetic operations."""
    ADD = 'add'
    SUBTRACT = 'subtract'
    MULTIPLY = 'multiply'
    DIVIDE = 'divide'
    UNKNOWN = 'unknown'


@dataclass
class StateUpdate:
    """Record of a state change for an entity."""
    timestep: int
    entity: str
    operation: OperationType
    operand: float
    old_value: float
    new_value: float
    source_text: str = ''


@dataclass
class Constraint:
    """Equation constraint from problem text."""
    lhs: str  # Left-hand side expression
    rhs: str  # Right-hand side expression
    relation: str = '=='  # Relation type (==, <, >, <=, >=)
    confidence: float = 1.0
    source_text: str = ''


class SignInferenceRules:
    """
    Context-aware sign inference for word problem operations.

    Handles the nuances of English verbs that indicate addition or subtraction,
    including modifiers that flip or confirm the base sign.

    Examples:
    - "gives away" -> subtract (from subject)
    - "gives 5 more" -> add (modifier "more" flips the base sign)
    - "receives" -> add (to subject)
    - "pays" -> subtract (from subject/payer)
    """

    # Base signs for verbs based on perspective (subject/object)
    VERB_BASE_SIGNS: Dict[str, Dict[str, str]] = {
        # Transfer verbs (perspective-dependent)
        'gives': {'subject': 'subtract', 'object': 'add'},
        'give': {'subject': 'subtract', 'object': 'add'},
        'gave': {'subject': 'subtract', 'object': 'add'},
        'receives': {'subject': 'add', 'object': 'subtract'},
        'receive': {'subject': 'add', 'object': 'subtract'},
        'received': {'subject': 'add', 'object': 'subtract'},
        'gets': {'subject': 'add', 'object': 'subtract'},
        'get': {'subject': 'add', 'object': 'subtract'},
        'got': {'subject': 'add', 'object': 'subtract'},
        'takes': {'subject': 'add', 'object': 'subtract'},
        'take': {'subject': 'add', 'object': 'subtract'},
        'took': {'subject': 'add', 'object': 'subtract'},

        # Payment verbs (subject pays, loses money)
        'pays': {'subject': 'subtract', 'object': 'add'},
        'pay': {'subject': 'subtract', 'object': 'add'},
        'paid': {'subject': 'subtract', 'object': 'add'},
        'spends': {'subject': 'subtract'},
        'spend': {'subject': 'subtract'},
        'spent': {'subject': 'subtract'},
        'buys': {'subject': 'subtract'},  # Money perspective
        'buy': {'subject': 'subtract'},
        'bought': {'subject': 'subtract'},

        # Earning/gaining verbs (subject gains)
        'earns': {'subject': 'add'},
        'earn': {'subject': 'add'},
        'earned': {'subject': 'add'},
        'gains': {'subject': 'add'},
        'gain': {'subject': 'add'},
        'gained': {'subject': 'add'},
        'wins': {'subject': 'add'},
        'win': {'subject': 'add'},
        'won': {'subject': 'add'},
        'finds': {'subject': 'add'},
        'find': {'subject': 'add'},
        'found': {'subject': 'add'},
        'collects': {'subject': 'add'},
        'collect': {'subject': 'add'},
        'collected': {'subject': 'add'},

        # Loss verbs (subject loses)
        'loses': {'subject': 'subtract'},
        'lose': {'subject': 'subtract'},
        'lost': {'subject': 'subtract'},
        'drops': {'subject': 'subtract'},
        'drop': {'subject': 'subtract'},
        'dropped': {'subject': 'subtract'},

        # Consumption verbs (subject uses up)
        'eats': {'subject': 'subtract'},
        'eat': {'subject': 'subtract'},
        'ate': {'subject': 'subtract'},
        'uses': {'subject': 'subtract'},
        'use': {'subject': 'subtract'},
        'used': {'subject': 'subtract'},
        'bakes': {'subject': 'subtract'},  # Uses ingredients
        'bake': {'subject': 'subtract'},
        'baked': {'subject': 'subtract'},
        'sells': {'subject': 'subtract'},  # Item perspective
        'sell': {'subject': 'subtract'},
        'sold': {'subject': 'subtract'},

        # Addition verbs
        'adds': {'subject': 'add'},
        'add': {'subject': 'add'},
        'added': {'subject': 'add'},
        'has': {'subject': 'set'},  # Initial value, not operation
        'have': {'subject': 'set'},
        'had': {'subject': 'set'},
    }

    # Modifiers that flip the base sign
    # "gives 5 more" -> recipient gets more, so if tracking recipient: add
    SIGN_FLIP_MODIFIERS = ['more', 'additional', 'extra', 'another', 'increases']

    # Modifiers that confirm subtraction
    SIGN_CONFIRM_SUBTRACT = ['away', 'off', 'out', 'back', 'fewer', 'less']

    # Modifiers that confirm addition
    SIGN_CONFIRM_ADD = ['together', 'total', 'combined', 'altogether']

    def infer(self, verb: str, context: Dict[str, Any]) -> str:
        """
        Infer whether an operation should add or subtract.

        Args:
            verb: The action verb (gives, earns, etc.)
            context: Dictionary containing:
                - subject: Who performs the action
                - object: Who receives (if applicable)
                - modifier: Words near the verb (more, away, etc.)
                - perspective: Whose state are we tracking? ('subject' or 'object')
                - surrounding_text: Full context for complex inference

        Returns:
            'add', 'subtract', 'set', or 'unknown'
        """
        verb_lower = verb.lower().strip()
        perspective = context.get('perspective', 'subject')
        modifier = context.get('modifier', '').lower()

        # Get base sign for verb+perspective
        verb_signs = self.VERB_BASE_SIGNS.get(verb_lower, {})
        base_sign = verb_signs.get(perspective, 'unknown')

        # If verb not found, try to infer from context
        if base_sign == 'unknown':
            base_sign = self._infer_from_context(verb_lower, context)

        # Check for sign flip modifiers
        # "gives 5 more" when tracking recipient -> add
        if any(m in modifier for m in self.SIGN_FLIP_MODIFIERS):
            if base_sign == 'subtract':
                return 'add'
            elif base_sign == 'add':
                return 'subtract'

        # Check for sign confirm modifiers
        if any(m in modifier for m in self.SIGN_CONFIRM_SUBTRACT):
            if base_sign in ['subtract', 'unknown']:
                return 'subtract'

        if any(m in modifier for m in self.SIGN_CONFIRM_ADD):
            if base_sign in ['add', 'unknown']:
                return 'add'

        return base_sign

    def _infer_from_context(self, verb: str, context: Dict[str, Any]) -> str:
        """Infer sign from surrounding context when verb not in dictionary."""
        surrounding = context.get('surrounding_text', '').lower()

        # Look for negative indicators
        negative_words = ['lose', 'lost', 'away', 'fewer', 'less', 'reduce',
                         'decrease', 'remove', 'take out', 'subtract']
        if any(neg in surrounding for neg in negative_words):
            return 'subtract'

        # Look for positive indicators
        positive_words = ['gain', 'add', 'increase', 'more', 'extra',
                         'total', 'together', 'plus']
        if any(pos in surrounding for pos in positive_words):
            return 'add'

        return 'unknown'

    def get_verb_info(self, verb: str) -> Dict[str, Any]:
        """Get information about a verb's sign behavior."""
        verb_lower = verb.lower().strip()
        info = {
            'verb': verb_lower,
            'known': verb_lower in self.VERB_BASE_SIGNS,
            'signs': self.VERB_BASE_SIGNS.get(verb_lower, {}),
            'perspective_dependent': len(self.VERB_BASE_SIGNS.get(verb_lower, {})) > 1
        }
        return info


class KnowledgeBase:
    """
    Semantic knowledge base for storing facts about entities and relationships.

    Stores:
    - Entity facts (John is a person, eggs is a food item)
    - Quantity associations (John has 5 eggs)
    - Relationships (John gives to Mary)
    - Constraints (total = sum of parts)
    """

    def __init__(self):
        self.facts: Dict[str, List[Any]] = {}
        self.entity_types: Dict[str, str] = {}
        self.relationships: List[Tuple[str, str, str]] = []  # (subject, relation, object)

    def add_fact(self, entity: str, attribute: str, value: Any):
        """Add a fact about an entity."""
        key = f"{entity}.{attribute}"
        if key not in self.facts:
            self.facts[key] = []
        self.facts[key].append(value)

    def get_fact(self, entity: str, attribute: str) -> Optional[Any]:
        """Get the most recent fact about an entity."""
        key = f"{entity}.{attribute}"
        if key in self.facts and self.facts[key]:
            return self.facts[key][-1]
        return None

    def add_relationship(self, subject: str, relation: str, obj: str):
        """Add a relationship between entities."""
        self.relationships.append((subject, relation, obj))

    def get_relationships(self, entity: str) -> List[Tuple[str, str, str]]:
        """Get all relationships involving an entity."""
        return [(s, r, o) for s, r, o in self.relationships
                if s == entity or o == entity]

    def set_entity_type(self, entity: str, entity_type: str):
        """Set the type of an entity."""
        self.entity_types[entity] = entity_type

    def get_entity_type(self, entity: str) -> Optional[str]:
        """Get the type of an entity."""
        return self.entity_types.get(entity)

    def clear(self):
        """Clear all stored knowledge."""
        self.facts.clear()
        self.entity_types.clear()
        self.relationships.clear()


class ReasoningTensor:
    """
    Modified NanoTensor for word problem reasoning.

    Core capabilities:
    1. Entity-Time State Matrix - Track entity values over narrative time
    2. Constraint Accumulator - Collect equations from text
    3. Semantic KB - Store facts about entities and relationships
    4. Sign Inference Engine - Determine operation polarity

    Unlike NanoTensor which focuses on symbolic tensor operations and
    Taylor expansions, ReasoningTensor focuses on:
    - Tracking discrete state changes over a narrative
    - Resolving comparative relationships
    - Inferring operation signs from linguistic context
    """

    def __init__(self, entities: List[str] = None, max_timesteps: int = 10):
        """
        Initialize ReasoningTensor.

        Args:
            entities: List of entity names to track (can be empty initially)
            max_timesteps: Maximum number of state changes to track
        """
        entities = entities or []
        self.max_timesteps = max_timesteps

        # Entity name -> index mapping
        self.entities: Dict[str, int] = {name: idx for idx, name in enumerate(entities)}

        # Entity × Timestep state matrix (values can be None for uninitialized)
        self._state_matrix: np.ndarray = np.empty(
            (max(len(entities), 1), max_timesteps),
            dtype=object
        )
        self._state_matrix.fill(None)

        # Current timestep for each entity
        self._current_timesteps: Dict[str, int] = {name: 0 for name in entities}
        self.global_timestep = 0

        # History of all state updates
        self.state_history: List[StateUpdate] = []

        # Constraint accumulator
        self.constraints: List[Constraint] = []

        # Semantic knowledge base
        self.kb = KnowledgeBase()

        # Sign inference rules
        self.sign_rules = SignInferenceRules()

    def add_entity(self, entity: str) -> int:
        """
        Add a new entity to track.

        Args:
            entity: Name of the entity

        Returns:
            Index of the entity in the state matrix
        """
        if entity in self.entities:
            return self.entities[entity]

        idx = len(self.entities)
        self.entities[entity] = idx
        self._current_timesteps[entity] = 0

        # Expand state matrix if needed
        if idx >= self._state_matrix.shape[0]:
            new_matrix = np.empty(
                (idx + 1, self.max_timesteps),
                dtype=object
            )
            new_matrix.fill(None)
            new_matrix[:self._state_matrix.shape[0], :] = self._state_matrix
            self._state_matrix = new_matrix

        return idx

    def set_entity_value(self, entity: str, value: float,
                         timestep: int = None, source_text: str = ''):
        """
        Set entity value at a specific timestep.

        Args:
            entity: Entity name
            value: Numeric value to set
            timestep: Optional timestep (uses current if None)
            source_text: Original text this value came from
        """
        if entity not in self.entities:
            self.add_entity(entity)

        idx = self.entities[entity]
        t = timestep if timestep is not None else self._current_timesteps.get(entity, 0)

        # Record the update
        old_value = self._state_matrix[idx, t] if t < self.max_timesteps else None
        self._state_matrix[idx, t] = value

        # Store in knowledge base
        self.kb.add_fact(entity, 'value', value)

        # Record state update
        update = StateUpdate(
            timestep=t,
            entity=entity,
            operation=OperationType.ADD if old_value is None else OperationType.UNKNOWN,
            operand=value,
            old_value=old_value if old_value is not None else 0.0,
            new_value=value,
            source_text=source_text
        )
        self.state_history.append(update)

    def get_entity_value(self, entity: str, timestep: int = None) -> Optional[float]:
        """
        Get entity value at a specific timestep.

        Args:
            entity: Entity name
            timestep: Optional timestep (uses current if None)

        Returns:
            Value at timestep, or None if not set
        """
        if entity not in self.entities:
            return None

        idx = self.entities[entity]
        t = timestep if timestep is not None else self._current_timesteps.get(entity, 0)

        # Search backwards for most recent value if current is None
        if self._state_matrix[idx, t] is None:
            for check_t in range(t - 1, -1, -1):
                if self._state_matrix[idx, check_t] is not None:
                    return float(self._state_matrix[idx, check_t])
            return None

        return float(self._state_matrix[idx, t])

    def get_latest_value(self, entity: str) -> Optional[float]:
        """Get the most recent value for an entity."""
        if entity not in self.entities:
            return None

        idx = self.entities[entity]
        t = self._current_timesteps.get(entity, 0)

        # Search backwards for most recent non-None value
        for check_t in range(t, -1, -1):
            if self._state_matrix[idx, check_t] is not None:
                return float(self._state_matrix[idx, check_t])
        return None

    def apply_operation(self, entity: str, operation: Union[str, OperationType],
                        operand: float, source_text: str = '') -> float:
        """
        Apply an operation to an entity and advance its timestep.

        Args:
            entity: Entity name
            operation: 'add', 'subtract', 'multiply', 'divide' or OperationType
            operand: Value to use in operation
            source_text: Original text this operation came from

        Returns:
            New value after operation
        """
        if entity not in self.entities:
            self.add_entity(entity)

        # Normalize operation type
        if isinstance(operation, str):
            op_map = {
                'add': OperationType.ADD,
                'subtract': OperationType.SUBTRACT,
                'multiply': OperationType.MULTIPLY,
                'divide': OperationType.DIVIDE,
            }
            operation = op_map.get(operation.lower(), OperationType.UNKNOWN)

        # Get current value
        current_val = self.get_latest_value(entity)
        if current_val is None:
            current_val = 0.0

        # Apply operation
        if operation == OperationType.ADD:
            new_val = current_val + operand
        elif operation == OperationType.SUBTRACT:
            new_val = current_val - operand
        elif operation == OperationType.MULTIPLY:
            new_val = current_val * operand
        elif operation == OperationType.DIVIDE:
            new_val = current_val / operand if operand != 0 else float('inf')
        else:
            new_val = current_val

        # Advance timestep and set new value
        current_t = self._current_timesteps.get(entity, 0)
        new_t = min(current_t + 1, self.max_timesteps - 1)
        self._current_timesteps[entity] = new_t
        self._state_matrix[self.entities[entity], new_t] = new_val
        self.global_timestep = max(self.global_timestep, new_t)

        # Record state update
        update = StateUpdate(
            timestep=new_t,
            entity=entity,
            operation=operation,
            operand=operand,
            old_value=current_val,
            new_value=new_val,
            source_text=source_text
        )
        self.state_history.append(update)

        # Update KB
        self.kb.add_fact(entity, 'value', new_val)

        return new_val

    def infer_operation_sign(self, verb: str, context: Dict[str, Any]) -> str:
        """
        Infer whether an operation should add or subtract.

        Args:
            verb: The action verb (gives, earns, etc.)
            context: Dictionary with subject, object, modifier, perspective, etc.

        Returns:
            'add', 'subtract', 'set', or 'unknown'
        """
        return self.sign_rules.infer(verb, context)

    def add_constraint(self, lhs: str, rhs: str, relation: str = '==',
                       confidence: float = 1.0, source_text: str = ''):
        """
        Add an equation constraint.

        Args:
            lhs: Left-hand side expression (e.g., 'john')
            rhs: Right-hand side expression (e.g., '2 * mary')
            relation: Relation type ('==', '<', '>', '<=', '>=')
            confidence: Confidence in this constraint
            source_text: Original text this constraint came from
        """
        constraint = Constraint(
            lhs=lhs,
            rhs=rhs,
            relation=relation,
            confidence=confidence,
            source_text=source_text
        )
        self.constraints.append(constraint)

    def solve_constraints(self) -> Dict[str, float]:
        """
        Solve accumulated constraints.

        Uses simple substitution and linear algebra for supported systems.

        Returns:
            Dictionary mapping variable names to solved values
        """
        if not self.constraints:
            return {}

        # Build known values from entity states
        known: Dict[str, float] = {}
        for entity in self.entities:
            val = self.get_latest_value(entity)
            if val is not None:
                known[entity.lower()] = val

        # Try to solve constraints iteratively
        max_iterations = 10
        for _ in range(max_iterations):
            progress = False

            for constraint in self.constraints:
                if constraint.relation != '==':
                    continue

                lhs = constraint.lhs.lower()
                rhs = constraint.rhs.lower()

                # Try to evaluate RHS and solve for LHS
                if lhs not in known:
                    try:
                        rhs_value = self._evaluate_expression(rhs, known)
                        if rhs_value is not None:
                            known[lhs] = rhs_value
                            progress = True
                    except Exception:
                        pass

                # Try the reverse
                if lhs in known and rhs not in known:
                    # If RHS is a simple variable
                    if rhs.isalpha():
                        known[rhs] = known[lhs]
                        progress = True

            if not progress:
                break

        return known

    def _evaluate_expression(self, expr: str, variables: Dict[str, float]) -> Optional[float]:
        """
        Evaluate a simple arithmetic expression with known variables.

        Supports: +, -, *, /, numbers, and variable names
        """
        try:
            # Substitute known values
            for var, val in variables.items():
                expr = expr.replace(var, str(val))

            # Evaluate safely
            allowed = {
                '__builtins__': {},
                'abs': abs,
            }
            result = eval(expr, allowed, {})
            return float(result)
        except Exception:
            return None

    def get_final_states(self) -> Dict[str, float]:
        """Get the final state values for all entities."""
        states = {}
        for entity in self.entities:
            val = self.get_latest_value(entity)
            if val is not None:
                states[entity] = val
        return states

    def get_state_history(self, entity: str = None) -> List[StateUpdate]:
        """
        Get state update history.

        Args:
            entity: Optional entity name to filter by

        Returns:
            List of state updates, optionally filtered
        """
        if entity is None:
            return self.state_history.copy()
        return [u for u in self.state_history if u.entity == entity]

    def reset(self):
        """Reset all state tracking."""
        self._state_matrix.fill(None)
        for entity in self.entities:
            self._current_timesteps[entity] = 0
        self.global_timestep = 0
        self.state_history.clear()
        self.constraints.clear()
        self.kb.clear()

    def __repr__(self) -> str:
        return (f"ReasoningTensor(entities={list(self.entities.keys())}, "
                f"timestep={self.global_timestep}, "
                f"constraints={len(self.constraints)})")
