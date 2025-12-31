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

import ast
import logging
import re
from typing import Any, Dict, List, Optional


class SymboBridge:
    """
    Lightweight integration layer that wires Symbo components into the
    Algorithm-Tumbler stack while degrading gracefully when optional
    dependencies are unavailable.
    """

    def __init__(self, enable_llm: bool = True, enable_tensor: bool = True):
        self.logger = logging.getLogger(__name__)

        # Symbo LLM
        self._llm_task_cls = None
        self.llm_adapter = None
        if enable_llm:
            try:
                from symbo_llm import LLMTask, SymboLLMAdapter

                self._llm_task_cls = LLMTask
                self.llm_adapter = SymboLLMAdapter(
                    checkpoint_dir="data/symbo_llm/unified"
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Symbo LLM unavailable: %s", exc)

        # Tensor engines
        self._nano_module = None
        self._reasoning_module = None
        if enable_tensor:
            try:
                import nano_tensor as nano_module

                self._nano_module = nano_module
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("NanoTensor unavailable: %s", exc)

            try:
                import reasoning_tensor as reasoning_module

                self._reasoning_module = reasoning_module
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("ReasoningTensor unavailable: %s", exc)

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    def summarize_code(self, code: str, context: Optional[str] = None) -> str:
        """Summarize code using SymboLLM when available, else fallback."""
        if self.llm_adapter and self._llm_task_cls:
            task = self._llm_task_cls(
                prompt=f"Summarize the following code:\n{code}",
                context=context,
                task_type="reasoning",
                max_tokens=128,
                temperature=0.2,
            )
            response = self.llm_adapter.handle_task(task)
            if response:
                return f"SymboLLM summary: {response}"

        snippet = (code or "").strip().replace("\n", " ")
        snippet = snippet[:80] + ("..." if len(snippet) > 80 else "")
        return f"Symbo summary (fallback): {snippet}"

    def simplify_expression(self, expression: str) -> Dict[str, Any]:
        """
        Simplify/evaluate a mathematical expression using NanoTensor if
        available, otherwise a safe AST evaluator.
        """
        # Attempt NanoTensor path
        if self._nano_module and hasattr(self._nano_module, "parse_expr"):
            try:
                expr = self._nano_module.parse_expr(expression)
                simplified = self._nano_module.simplify(expr)
                return {
                    "engine": "nano_tensor",
                    "result": str(simplified),
                }
            except Exception:
                # Fall back to safe evaluator
                pass

        value = self._safe_eval(expression)
        return {"engine": "safe_eval", "result": value}

    def reason_about_story(self, text: str) -> Dict[str, Any]:
        """
        Use ReasoningTensor to track entities if available; otherwise,
        produce a lightweight heuristic analysis.
        """
        if self._reasoning_module and hasattr(self._reasoning_module, "ReasoningTensor"):
            try:
                rt = self._reasoning_module.ReasoningTensor()
                # Basic extraction: numbers imply entities "value_<i>"
                numbers = self._extract_numbers(text)
                for idx, number in enumerate(numbers):
                    entity = f"value_{idx}"
                    rt.set_entity_value(entity, number, source_text=text)
                latest = {
                    ent: rt.get_latest_value(ent) for ent in rt.entities.keys()
                }
                return {
                    "engine": "reasoning_tensor",
                    "entities": latest,
                    "constraints": len(rt.constraints),
                }
            except Exception:
                # Fall back to heuristic path
                pass

        numbers = self._extract_numbers(text)
        return {
            "engine": "heuristic",
            "entities": {f"value_{i}": n for i, n in enumerate(numbers)},
            "constraints": 0,
        }

    def status(self) -> Dict[str, Any]:
        """Expose availability of Symbo components."""
        return {
            "llm_available": self.llm_adapter is not None,
            "nano_tensor_available": self._nano_module is not None,
            "reasoning_tensor_available": self._reasoning_module is not None,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _safe_eval(self, expression: str) -> Any:
        """Safely evaluate basic arithmetic expressions using AST."""
        try:
            node = ast.parse(expression, mode="eval")
            return self._eval_node(node.body)
        except Exception:
            return None

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant type")
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
        ):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._apply_op(node.op, left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._eval_node(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        raise ValueError("Unsupported expression")

    @staticmethod
    def _apply_op(op, left, right):
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.Pow):
            return left ** right
        if isinstance(op, ast.Mod):
            return left % right
        raise ValueError("Unsupported operator")

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        return [float(num) for num in re.findall(r"-?\d+(?:\.\d+)?", text or "")]
