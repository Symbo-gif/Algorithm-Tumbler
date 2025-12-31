# Copyright 2025 Damien Davison & Michael Maillet, Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NanoTensor and Symbolic Trainer for SYMBO_AGENTIC_REASONERS
=========================================

Core symbolic tensor operations with:
- Polynomial solving for exact solutions
- Perturbation methods for approximate solutions
- Taylor polynomial generation
- Hybrid symbolic-numeric training

Adapted for SYMBO_AGENTIC_REASONERS Phase 5 integration.

NO SYMPY - Uses native symbolic engine and numpy for all operations.
"""

import numpy as np
from functools import lru_cache
from typing import Tuple, Dict, Any, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Native symbolic imports - NO SYMPY
from symbo_agentic_reasoners.core.native_symbolic import (
    Symbol, sympify, simplify, parse_expr, diff as native_diff,
    symbols as create_symbols, expand as native_expand,
    Rational, Integer
)
from symbo_agentic_reasoners.core.calculus import (
    differentiate, solve_polynomial as native_solve
)

# Optional imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    # ImportError: torch not installed
    # RuntimeError: torch version incompatible with Python version (e.g., Python 3.14)
    # OSError: library loading issues
    TORCH_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from kanren import Relation, facts, run, var as kvar
    KANREN_AVAILABLE = True
except ImportError:
    KANREN_AVAILABLE = False

try:
    from skopt import gp_minimize
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for file export
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Helper functions that replace SymPy functionality
def _create_symbol(name: str) -> Symbol:
    """Create a symbolic variable."""
    return Symbol(name)


def _sympify_value(val):
    """Convert value to symbolic form if needed."""
    if isinstance(val, (int, float)):
        return Rational(val) if isinstance(val, int) else sympify(str(val))
    return sympify(str(val)) if not hasattr(val, 'free_symbols') else val


def _diff_expr(expr, wrt, order: int = 1):
    """Differentiate expression using native calculus."""
    for _ in range(order):
        success, result_str, _ = differentiate(str(expr), str(wrt))
        if success:
            expr = sympify(result_str)
        else:
            # Fallback to native_diff
            expr = native_diff(expr, wrt)
    return expr


def _subs_expr(expr, sub_dict: Dict):
    """Substitute values in expression."""
    if hasattr(expr, 'subs'):
        return expr.subs(sub_dict)
    # Manual substitution for simple cases
    result_str = str(expr)
    for k, v in sub_dict.items():
        result_str = result_str.replace(str(k), f"({v})")
    return sympify(result_str)


def _solve_polynomial(coefficients: List[float], var_name: str = 'x') -> List[float]:
    """
    Solve polynomial equation using numpy roots (for numerical solutions).

    Args:
        coefficients: Polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
                     representing a_n*x^n + ... + a_1*x + a_0 = 0
        var_name: Variable name (for documentation)

    Returns:
        List of real roots
    """
    try:
        roots = np.roots(coefficients)
        # Filter for real roots
        real_roots = []
        for r in roots:
            if abs(np.imag(r)) < 1e-10:
                real_roots.append(float(np.real(r)))
        return real_roots
    except Exception:
        return []


def _lambdify_native(vars_list, expr):
    """
    Create a numerical function from symbolic expression.

    Returns a callable that evaluates the expression numerically.
    """
    expr_str = str(expr)
    var_names = [str(v) for v in vars_list]

    def evaluator(*args):
        """Perform evaluator operation.

        Args:
        No arguments

        Returns:
        Result of the operation

        Example:
        >>> result = obj.evaluator(...)
        """
        result_str = expr_str
        for name, val in zip(var_names, args):
            result_str = result_str.replace(name, str(val))
        try:
            # Use eval with numpy functions available
            import math
            safe_dict = {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'abs': np.abs, 'pi': np.pi, 'e': np.e,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            }
            return eval(result_str, {"__builtins__": {}}, safe_dict)
        except Exception:
            return float('nan')

    return evaluator


def _latex_native(expr) -> str:
    """Convert expression to LaTeX string."""
    expr_str = str(expr)
    # Basic LaTeX conversions
    expr_str = expr_str.replace('**', '^')
    expr_str = expr_str.replace('*', ' \\cdot ')
    expr_str = expr_str.replace('sqrt', '\\sqrt')
    expr_str = expr_str.replace('sin', '\\sin')
    expr_str = expr_str.replace('cos', '\\cos')
    expr_str = expr_str.replace('tan', '\\tan')
    expr_str = expr_str.replace('exp', '\\exp')
    expr_str = expr_str.replace('log', '\\log')
    return expr_str


class NanoTensor:
    """
    True nD symbolic tensor with vectorized operations, polynomial solving,
    and full perturbation capabilities. Represents mathematical objects exactly.

    NO SYMPY - Uses native symbolic engine for all operations.
    """

    def __init__(self, shape: Tuple[int, ...], max_order: int = 2,
                 base_vars: List[str] = None, name: str = "nt"):
        self.shape = shape
        self.max_order = max_order
        self.name = name
        self.base_vars = [_create_symbol(v) for v in (base_vars or ['k', 'a', 'eps', 'sig'])]
        self.coeff_vars: List[Symbol] = []
        self.data: np.ndarray = np.empty(shape, dtype=object)
        self._init_data()
        self._symvars_cache = None
        self._lambdify_cache: Dict = {}
        self.fitted_coeffs: Dict[str, float] = {}

    def _init_data(self):
        """Initialize tensor with symbolic zeros"""
        self.data = np.zeros(self.shape, dtype=object)
        self.data.flat[:] = Rational(0)

    @property
    def symvars(self) -> List[Symbol]:
        """Cache all free symbols in the tensor"""
        if self._symvars_cache is None:
            vars_set = set()
            for elem in self.data.flat:
                if elem != 0 and hasattr(elem, 'free_symbols'):
                    vars_set.update(elem.free_symbols)
            self._symvars_cache = list(vars_set)
        return self._symvars_cache

    @lru_cache(maxsize=128)
    def diff_cached(self, wrt_name: str, order: int = 1) -> 'NanoTensor':
        """Cached differentiation for performance"""
        wrt_candidates = [v for v in self.symvars if str(v) == wrt_name]
        if not wrt_candidates:
            wrt_candidates = [v for v in self.base_vars if str(v) == wrt_name]
        if not wrt_candidates:
            raise ValueError(f"Variable {wrt_name} not found")
        return self.diff(wrt_candidates[0], order)

    def diff(self, wrt: Symbol, order: int = 1) -> 'NanoTensor':
        """Symbolic differentiation of entire tensor using native calculus"""
        new_nt = NanoTensor(self.shape, self.max_order, [str(v) for v in self.base_vars])
        new_nt.data = np.vectorize(lambda e: _diff_expr(e, wrt, order))(self.data)
        return new_nt

    def subs(self, sub_dict: Dict[Symbol, Any]) -> 'NanoTensor':
        """Symbolic substitution across tensor"""
        new_nt = NanoTensor(self.shape, self.max_order, [str(v) for v in self.base_vars])
        clean_dict = {k: (_sympify_value(v) if isinstance(v, (int, float)) else v)
                     for k, v in sub_dict.items()}
        new_nt.data = np.vectorize(lambda e: _subs_expr(e, clean_dict))(self.data)
        return new_nt

    @lru_cache(maxsize=128)
    def subs_cached(self, sub_tuple: Tuple) -> 'NanoTensor':
        """Cached substitution for repeated calls"""
        sub_dict = dict(sub_tuple)
        return self.subs(sub_dict)

    def eval_numeric(self, point: Dict[str, float], use_lambdify: bool = True) -> np.ndarray:
        """Evaluate tensor numerically at a given point with caching."""
        vars_in_point = [v for v in self.symvars if str(v) in point]
        key = tuple(sorted([str(v) for v in vars_in_point])) + tuple(sorted(point.keys()))
        if key not in self._lambdify_cache:
            self._lambdify_cache[key] = [
                _lambdify_native(vars_in_point, e) for e in self.data.flat
            ]
        funcs = self._lambdify_cache[key]
        args = [point.get(str(v), 0.0) for v in vars_in_point]
        result_flat = [f(*args) for f in funcs]
        return np.array(result_flat).reshape(self.shape)

    def solve_poly(self, target_eq, var) -> List[float]:
        """Solve polynomial equation and return real roots using numpy."""
        try:
            # Simplify and convert to string for parsing
            eq_str = str(simplify(target_eq))
            var_str = str(var)

            # Try native solve first
            solutions = native_solve(eq_str, var_str)
            if solutions:
                real_roots = []
                for sol in solutions:
                    try:
                        val = float(str(sol))
                        if isinstance(val, complex):
                            if abs(val.imag) < 1e-8:
                                real_roots.append(float(val.real))
                        else:
                            real_roots.append(val)
                    except (ValueError, TypeError):
                        pass
                return real_roots
            return []
        except Exception:
            return []

    def groebner_solve(self, poly_system: List,
                   vars_to_solve: List[Symbol] = None) -> List[Dict[str, Any]]:
        """
        Solve polynomial system using linear algebra for 2x2 systems.

        Note: True Groebner basis computation would require dedicated algorithm.
        This uses matrix-based solving for simple 2-variable linear systems.
        """
        if vars_to_solve is None:
            vars_to_solve = self.symvars[:len(poly_system)]

        try:
            # For 2-variable linear systems, use matrix method
            if len(poly_system) == 2 and len(vars_to_solve) == 2:
                var1_str = str(vars_to_solve[0])
                var2_str = str(vars_to_solve[1])

                # Extract coefficients for linear system: a1*x + b1*y + c1 = 0
                def extract_linear_coeffs(expr, var1, var2):
                    """Extract coefficients from linear expression."""
                    import re
                    expr_str = str(expr).replace(' ', '')

                    # Initialize coefficients
                    a, b, c = 0.0, 0.0, 0.0

                    # Parse terms
                    # Handle patterns like: -3+x+y or -1+x-1*y
                    terms = re.findall(r'[+-]?[^+-]+', expr_str)

                    for term in terms:
                        term = term.strip()
                        if not term:
                            continue

                        if var1 in term and var2 not in term:
                            # Coefficient of var1
                            coef_str = term.replace(var1, '').replace('*', '')
                            if coef_str in ['', '+']:
                                a = 1.0
                            elif coef_str == '-':
                                a = -1.0
                            else:
                                a = float(coef_str)
                        elif var2 in term and var1 not in term:
                            # Coefficient of var2
                            coef_str = term.replace(var2, '').replace('*', '')
                            if coef_str in ['', '+']:
                                b = 1.0
                            elif coef_str == '-':
                                b = -1.0
                            else:
                                b = float(coef_str)
                        elif var1 not in term and var2 not in term:
                            # Constant term
                            try:
                                c = float(term)
                            except ValueError:
                                pass

                    return a, b, c

                a1, b1, c1 = extract_linear_coeffs(poly_system[0], var1_str, var2_str)
                a2, b2, c2 = extract_linear_coeffs(poly_system[1], var1_str, var2_str)

                # Solve: a1*x + b1*y = -c1
                #        a2*x + b2*y = -c2
                det = a1 * b2 - a2 * b1
                if abs(det) > 1e-10:
                    x = (-c1 * b2 + c2 * b1) / det
                    y = (a1 * (-c2) - a2 * (-c1)) / det
                    return [{var1_str: x, var2_str: y}]

            # Fallback: try sequential solving for simpler systems
            solutions = []
            for eq in poly_system:
                eq_str = str(eq)
                for var in vars_to_solve:
                    var_str = str(var)
                    sols = native_solve(eq_str, var_str)
                    if sols:
                        for sol in sols:
                            try:
                                val = float(str(sol))
                                solutions.append({var_str: val})
                            except (ValueError, TypeError):
                                pass
            return solutions
        except Exception:
            return []

    def resultant(self, f, g, var) -> Any:
        """
        Compute resultant of two polynomials (simplified).

        The resultant is 0 iff the polynomials share a common root.
        Full resultant computation requires Sylvester matrix determinant.
        This simplified version checks for common roots numerically.
        """
        try:
            f_str = str(f).replace('^', '**')
            g_str = str(g).replace('^', '**')
            var_str = str(var)

            # For simple cases like x^2 - 1 and x - 1:
            # Try to find roots numerically using numpy
            try:
                import numpy as np

                # Create evaluation functions
                def eval_f(x_val):
                    """Perform eval f operation.

                    Args:
                    x_val: Description needed

                    Returns:
                    Result of the operation

                    Example:
                    >>> result = obj.eval_f(...)
                    """
                    return eval(f_str.replace(var_str, str(x_val)))

                def eval_g(x_val):
                    """Perform eval g operation.

                    Args:
                    x_val: Description needed

                    Returns:
                    Result of the operation

                    Example:
                    >>> result = obj.eval_g(...)
                    """
                    return eval(g_str.replace(var_str, str(x_val)))

                # Search for roots of g (simpler polynomial typically)
                test_points = np.linspace(-10, 10, 100)
                g_roots = []

                # Find sign changes for g
                for i in range(len(test_points) - 1):
                    try:
                        g1 = eval_g(test_points[i])
                        g2 = eval_g(test_points[i+1])
                        if g1 * g2 < 0:
                            # Sign change - root between these points
                            # Use bisection to find root
                            a, b = test_points[i], test_points[i+1]
                            for _ in range(50):
                                mid = (a + b) / 2
                                if eval_g(mid) * eval_g(a) < 0:
                                    b = mid
                                else:
                                    a = mid
                            g_roots.append((a + b) / 2)
                    except Exception:
                        pass

                # Also check for exact integer roots of g
                for x_val in range(-10, 11):
                    try:
                        if abs(eval_g(x_val)) < 1e-10:
                            g_roots.append(x_val)
                    except Exception:
                        pass

                # Check if any root of g is also a root of f
                for root in g_roots:
                    try:
                        f_val = eval_f(root)
                        if abs(f_val) < 1e-8:
                            return Integer(0)
                    except Exception:
                        pass

            except ImportError:
                # Fallback without numpy
                for x_val in range(-10, 11):
                    try:
                        g_val = eval(g_str.replace(var_str, str(x_val)))
                        if abs(g_val) < 1e-10:
                            f_val = eval(f_str.replace(var_str, str(x_val)))
                            if abs(f_val) < 1e-10:
                                return Integer(0)
                    except Exception:
                        pass

            # No common root found
            return Integer(1)
        except Exception:
            return Integer(0)

    def parametrize_curve(self, implicit_poly,
                          t = None) -> Tuple[Any, Any]:
        """Parametrize algebraic curve (simplified approach)."""
        if t is None:
            t = _create_symbol('t')
        x = _create_symbol('x')
        y = _create_symbol('y')

        implicit_poly = sympify(str(implicit_poly))

        # Simplified parametrization - line intersection approach
        try:
            # Substitute y = t*x
            expr_str = str(implicit_poly)
            subst_str = expr_str.replace(str(y), f"({t}*{x})")
            subst_expr = sympify(subst_str)

            # Solve for x
            x_sols = native_solve(str(subst_expr), str(x))
            if x_sols:
                non_trivial = [sol for sol in x_sols if str(sol) != '0']
                if non_trivial:
                    x_param = simplify(sympify(str(non_trivial[-1])))
                    y_param = simplify(sympify(f"({t})*({x_param})"))
                    return x_param, y_param
        except Exception:
            pass

        return x, y

    def generate_taylor(self, center: Dict[str, float], ss_value = None):
        """Generate Taylor polynomial up to max_order, including cross terms."""
        self.coeff_vars.clear()
        if ss_value is None:
            ss_value = sum([v for v in self.base_vars])

        devs = [_create_symbol(str(v)) - center.get(str(v), 0) for v in self.base_vars]
        taylor_expr = ss_value

        self.coeff_vars.clear()

        # Linear terms
        for v, dev in zip(self.base_vars, devs):
            c = _create_symbol(f'g_{str(v)}')
            self.coeff_vars.append(c)
            taylor_expr = sympify(f"({taylor_expr}) + ({c})*({dev})")

        # Quadratic and interaction terms
        if self.max_order >= 2:
            for i, v1 in enumerate(self.base_vars):
                c = _create_symbol(f'g_{str(v1)}_{str(v1)}')
                self.coeff_vars.append(c)
                taylor_expr = sympify(f"({taylor_expr}) + 0.5*({c})*({devs[i]})**2")
                for j in range(i+1, len(self.base_vars)):
                    v2 = self.base_vars[j]
                    c_cross = _create_symbol(f'g_{str(v1)}_{str(v2)}')
                    self.coeff_vars.append(c_cross)
                    taylor_expr = sympify(f"({taylor_expr}) + ({c_cross})*({devs[i]})*({devs[j]})")

        self.data.flat[:] = simplify(taylor_expr)

    def compute_steady_state(self, model_eqs: List,
                            params: Dict[str, float],
                            ss_guess: Dict[str, float] = None) -> Dict[str, float]:
        """Compute steady state by solving model equations."""
        if ss_guess is None:
            ss_guess = {str(v): 1.0 for v in self.base_vars}

        # Substitute parameters into equations
        eqs_ss = []
        for eq in model_eqs:
            eq_str = str(eq)
            for k, v in params.items():
                eq_str = eq_str.replace(k, str(v))
            eqs_ss.append(eq_str)

        # Try native solve for each equation
        try:
            for var_name in ss_guess.keys():
                for eq_str in eqs_ss:
                    sols = native_solve(eq_str, var_name)
                    if sols:
                        try:
                            ss_guess[var_name] = float(str(sols[0]))
                        except (ValueError, TypeError):
                            pass
            return ss_guess
        except Exception:
            return ss_guess

    def full_perturbation(self, model_R, params: Dict[str, Any],
                         var_order: List[str] = ['k', 'a', 'eps', 'sig'],
                         eps_var: float = 1.0):
        """Full 2nd-order perturbation solver using native engine."""
        ss = self.compute_steady_state([model_R], params)
        self.coeff_vars.clear()
        self.generate_taylor(ss)
        self.fitted_coeffs = {}

        sym_vars = {str(v): v for v in self.base_vars}

        # First-order conditions
        for vname in var_order:
            if vname not in sym_vars:
                continue
            wrt = sym_vars[vname]

            # Compute first derivative
            R_v = _diff_expr(model_R, wrt, 1)
            R_v = simplify(R_v)

            # Substitute steady state and parameters
            R_v_ss = R_v
            for k, v in ss.items():
                R_v_ss = _subs_expr(R_v_ss, {_create_symbol(k): v})
            for k, v in params.items():
                R_v_ss = _subs_expr(R_v_ss, {_create_symbol(k): v})
            for k, v in self.fitted_coeffs.items():
                R_v_ss = _subs_expr(R_v_ss, {_create_symbol(k): v})

            coeff_name = f'g_{vname}'
            coeff_sym = _create_symbol(coeff_name)

            try:
                roots = self.solve_poly(R_v_ss, coeff_sym)
                if roots:
                    stable_root = min([r for r in roots if abs(r) < 10], key=abs, default=roots[0])
                    self.fitted_coeffs[coeff_name] = stable_root
            except Exception:
                pass

        # Second-order conditions
        if self.max_order >= 2:
            for i, v1 in enumerate(var_order):
                if v1 not in sym_vars:
                    continue
                wrt1 = sym_vars[v1]

                R_vv = _diff_expr(model_R, wrt1, 2)
                R_vv = simplify(R_vv)

                # Substitute
                R_vv_ss = R_vv
                for k, v in ss.items():
                    R_vv_ss = _subs_expr(R_vv_ss, {_create_symbol(k): v})
                for k, v in params.items():
                    R_vv_ss = _subs_expr(R_vv_ss, {_create_symbol(k): v})
                for k, v in self.fitted_coeffs.items():
                    R_vv_ss = _subs_expr(R_vv_ss, {_create_symbol(k): v})

                coeff_name = f'g_{v1}_{v1}'
                coeff_sym = _create_symbol(coeff_name)

                try:
                    roots = self.solve_poly(R_vv_ss, coeff_sym)
                    if roots:
                        self.fitted_coeffs[coeff_name] = roots[0]
                except Exception:
                    self.fitted_coeffs[coeff_name] = 0.0

                # Cross terms
                for j in range(i+1, len(var_order)):
                    v2 = var_order[j]
                    if v2 not in sym_vars:
                        continue
                    wrt2 = sym_vars[v2]

                    R_v1v2 = _diff_expr(_diff_expr(model_R, wrt1, 1), wrt2, 1)
                    R_v1v2 = simplify(R_v1v2)

                    R_v1v2_ss = R_v1v2
                    for k, v in ss.items():
                        R_v1v2_ss = _subs_expr(R_v1v2_ss, {_create_symbol(k): v})
                    for k, v in params.items():
                        R_v1v2_ss = _subs_expr(R_v1v2_ss, {_create_symbol(k): v})
                    for k, v in self.fitted_coeffs.items():
                        R_v1v2_ss = _subs_expr(R_v1v2_ss, {_create_symbol(k): v})

                    coeff_name = f'g_{v1}_{v2}'
                    coeff_sym = _create_symbol(coeff_name)

                    try:
                        roots = self.solve_poly(R_v1v2_ss, coeff_sym)
                        if roots:
                            self.fitted_coeffs[coeff_name] = roots[0]
                    except Exception:
                        self.fitted_coeffs[coeff_name] = 0.0

        # Apply solution to tensor
        apply_dict = {_create_symbol(k): v for k, v in self.fitted_coeffs.items()}
        self.data.flat[:] = self.subs(apply_dict).data.flat[0]

    def simplify_tensor(self):
        """Simplify all expressions in tensor using native simplify."""
        self.data = np.vectorize(lambda e: simplify(e))(self.data)
        self._symvars_cache = None

    def simplify(self):
        """Alias for simplify_tensor() for compatibility."""
        return self.simplify_tensor()

    def deriv_tree(self, wrt_vars: List[str]) -> Dict[str, Any]:
        """Explainability: Show derivative structure"""
        tree = {}
        for var_name in wrt_vars:
            try:
                deriv = self.diff_cached(var_name, 1).data.flat[0]
                tree[var_name] = simplify(deriv)
            except (KeyError, IndexError, TypeError, AttributeError):
                tree[var_name] = _create_symbol('NA')
        return tree

    def plot_contour(
        self,
        var1: str,
        var2: str,
        fixed_values: Dict[str, float] = None,
        output_path: str = None,
        n_points: int = 50,
        levels: int = 20,
        title: str = None,
        colormap: str = 'viridis'
    ) -> Optional[str]:
        """
        Export contour plot of tensor expression to file.

        Args:
            var1: Name of first variable (x-axis)
            var2: Name of second variable (y-axis)
            fixed_values: Dict of fixed values for other variables
            output_path: Path to save the plot (default: auto-generated)
            n_points: Number of points per axis
            levels: Number of contour levels
            title: Plot title (default: auto-generated)
            colormap: Matplotlib colormap name

        Returns:
            Path to saved file, or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("matplotlib not available - cannot generate contour plot")
            return None

        fixed_values = fixed_values or {}

        # Find the variables
        var1_sym = None
        var2_sym = None
        for v in self.base_vars + self.symvars:
            if str(v) == var1:
                var1_sym = v
            if str(v) == var2:
                var2_sym = v

        if var1_sym is None or var2_sym is None:
            raise ValueError(f"Variables {var1} or {var2} not found in tensor")

        # Get the expression (first element for scalar tensor)
        expr = self.data.flat[0]

        # Substitute fixed values
        for k, v in fixed_values.items():
            expr = _subs_expr(expr, {_create_symbol(k): v})

        # Create native lambdified function
        f = _lambdify_native([var1_sym, var2_sym], expr)

        # Create grid
        x1_range = np.linspace(-2, 2, n_points)
        x2_range = np.linspace(-2, 2, n_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        # Evaluate using vectorized native function
        try:
            Z = np.vectorize(f)(X1, X2)
            if not isinstance(Z, np.ndarray):
                Z = np.full_like(X1, float(Z))
        except Exception as e:
            warnings.warn(f"Could not evaluate expression: {e}")
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(X1, X2, Z, levels=levels, cmap=colormap)
        ax.contour(X1, X2, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.3)
        plt.colorbar(contour, ax=ax, label='Value')
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_title(title or f'Contour Plot: {self.name}({var1}, {var2})')

        # Generate output path if not provided
        if output_path is None:
            import tempfile
            output_path = tempfile.mktemp(suffix='_contour.png', prefix=f'{self.name}_')

        # Save
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def plot_surface(
        self,
        var1: str,
        var2: str,
        fixed_values: Dict[str, float] = None,
        output_path: str = None,
        n_points: int = 50,
        title: str = None,
        colormap: str = 'viridis',
        elevation: float = 30,
        azimuth: float = 45
    ) -> Optional[str]:
        """
        Export 3D surface plot of tensor expression to file.

        Args:
            var1: Name of first variable (x-axis)
            var2: Name of second variable (y-axis)
            fixed_values: Dict of fixed values for other variables
            output_path: Path to save the plot (default: auto-generated)
            n_points: Number of points per axis
            title: Plot title (default: auto-generated)
            colormap: Matplotlib colormap name
            elevation: Camera elevation angle
            azimuth: Camera azimuth angle

        Returns:
            Path to saved file, or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("matplotlib not available - cannot generate surface plot")
            return None

        fixed_values = fixed_values or {}

        # Find the variables
        var1_sym = None
        var2_sym = None
        for v in self.base_vars + self.symvars:
            if str(v) == var1:
                var1_sym = v
            if str(v) == var2:
                var2_sym = v

        if var1_sym is None or var2_sym is None:
            raise ValueError(f"Variables {var1} or {var2} not found in tensor")

        # Get the expression (first element for scalar tensor)
        expr = self.data.flat[0]

        # Substitute fixed values using native function
        for k, v in fixed_values.items():
            expr = _subs_expr(expr, {_create_symbol(k): v})

        # Create native lambdified function
        f = _lambdify_native([var1_sym, var2_sym], expr)

        # Create grid
        x1_range = np.linspace(-2, 2, n_points)
        x2_range = np.linspace(-2, 2, n_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        # Evaluate using vectorized native function
        try:
            Z = np.vectorize(f)(X1, X2)
            if not isinstance(Z, np.ndarray):
                Z = np.full_like(X1, float(Z))
        except Exception as e:
            warnings.warn(f"Could not evaluate expression: {e}")
            return None

        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X1, X2, Z, cmap=colormap, edgecolor='none', alpha=0.9)
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_zlabel('Value')
        ax.set_title(title or f'Surface Plot: {self.name}({var1}, {var2})')
        ax.view_init(elev=elevation, azim=azimuth)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Value')

        # Generate output path if not provided
        if output_path is None:
            import tempfile
            output_path = tempfile.mktemp(suffix='_surface.png', prefix=f'{self.name}_')

        # Save
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def export_expression_latex(self, output_path: str = None) -> str:
        """
        Export tensor expressions to LaTeX file for debugging.

        Args:
            output_path: Path to save LaTeX (default: auto-generated)

        Returns:
            LaTeX string representation
        """
        latex_parts = []
        latex_parts.append(f"% NanoTensor: {self.name}")
        latex_parts.append(f"% Shape: {self.shape}")
        latex_parts.append(f"% Base variables: {[str(v) for v in self.base_vars]}")
        latex_parts.append("")

        for idx, elem in enumerate(self.data.flat):
            latex_expr = _latex_native(elem)
            latex_parts.append(f"% Element [{idx}]:")
            latex_parts.append(f"$$ {latex_expr} $$")
            latex_parts.append("")

        if self.fitted_coeffs:
            latex_parts.append("% Fitted coefficients:")
            for k, v in self.fitted_coeffs.items():
                latex_parts.append(f"% {k} = {v:.6f}")

        latex_content = "\n".join(latex_parts)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)

        return latex_content


class SymbolicTrainer:
    """
    Trainer for symbolic tensors with multiple fitting algorithms:
    - 'symbolic': Gröbner basis exact solving
    - 'perturbation': 2nd-order perturbation method
    - 'lsq': Least squares numeric
    """

    def __init__(self, nano_tensor: NanoTensor, tolerance: float = 1e-8):
        self.nt = nano_tensor
        self.tolerance = tolerance
        self.fitted_coeffs: Dict[str, float] = {}
        self.kb = KnowledgeBase()

    def fit(self, data: List[Any], method: str = 'symbolic') -> bool:
        """
        Fit tensor to data using specified method.
        For perturbation: data = [(model_R, params_dict)]
        For symbolic/lsq: data = [(state_dict, target_value), ...]
        """
        if method == 'symbolic':
            return self._fit_symbolic(data)
        elif method == 'perturbation':
            if len(data) >= 2:
                model_R, params = data[0], data[1]
                self.nt.full_perturbation(model_R, params)
                self.fitted_coeffs = getattr(self.nt, 'fitted_coeffs', {})
                return len(self.fitted_coeffs) > 0
        elif method == 'lsq':
            return self._fit_lsq(data)

        return False

    def _fit_symbolic(self, data: List[Tuple[Dict[str, float], float]]) -> bool:
        """Exact fitting via polynomial solving"""
        equations = []
        coeffs_sym = self.nt.coeff_vars

        if not coeffs_sym:
            return False

        for state_point, target in data:
            subs_d = {self.nt.base_vars[i]: state_point.get(str(v), 0)
                     for i, v in enumerate(self.nt.base_vars)}
            nt_val = sum(self.nt.subs(subs_d).data.flat)
            # Create equation as string: nt_val - target = 0
            eq_str = f"({nt_val}) - ({target})"
            equations.append(sympify(eq_str))

        sols = self.nt.groebner_solve(equations, coeffs_sym)
        if sols:
            sol_dict = {}
            for k, v in sols[0].items():
                try:
                    sol_dict[k] = float(str(v))
                except (ValueError, TypeError):
                    sol_dict[k] = 0.0
            self._apply_solution(sol_dict)
            return True
        return False

    def _fit_lsq(self, data: List[Tuple[Dict[str, float], float]]) -> bool:
        """Least squares numeric fitting"""
        A, b = [], []
        coeffs_sym = self.nt.coeff_vars

        for state_point, target in data:
            row = []
            for coeff in coeffs_sym:
                basis = 0.0
                for i, v in enumerate(self.nt.base_vars):
                    if f'g_{str(v)}' == str(coeff):
                        basis += state_point.get(str(v), 0)
                row.append(float(basis))
            A.append(row)
            b.append(float(target))

        coeffs_num = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
        sol_dict = {str(c): coeffs_num[i] for i, c in enumerate(coeffs_sym)}
        self._apply_solution(sol_dict)
        return True

    def _apply_solution(self, sol_dict: Dict[str, float]):
        """Apply fitted coefficients to tensor and knowledge base"""
        subs_d = {_create_symbol(k): v for k, v in sol_dict.items()}
        self.nt = self.nt.subs(subs_d)
        self.fitted_coeffs = sol_dict

        for k, v in sol_dict.items():
            """Perform objective operation.

            Args:
            deg: Description needed

            Returns:
            Result of the operation

            Example:
            >>> result = obj.objective(...)
            """
            self.kb.add_fact('policy_coefficient', k, v)

    def predict(self, state_point: Dict[str, float]) -> np.ndarray:
        """Make prediction at given state"""
        return self.nt.eval_numeric(state_point)


class HybridTrainer(SymbolicTrainer):
    """Hybrid trainer combining symbolic and numeric methods"""

    def symbolic_regression(self, X: np.ndarray, y: np.ndarray,
                           max_deg: int = 5) -> NanoTensor:
        """Symbolic regression via GP optimization."""
        if not SKOPT_AVAILABLE:
            # Fallback without optimization
            return NanoTensor((1,), max_order=2,
                             base_vars=[f'x{i}' for i in range(X.shape[1])])

        def objective(deg):
            nt_try = NanoTensor((1,), max_order=int(deg[0]),
                               base_vars=[f'x{i}' for i in range(X.shape[1])])
            nt_try.generate_taylor({f'x{i}': 0 for i in range(X.shape[1])})
            trainer_try = HybridTrainer(nt_try)

            data = []
            for j in range(len(y)):
                state = {f'x{i}': X[j,i] for i in range(X.shape[1])}
                data.append((state, y[j]))

            trainer_try.fit(data, method='lsq')
            pred = trainer_try.predict_batch(X)
            return np.mean((pred - y)**2)

        res = gp_minimize(objective, [(-2, max_deg)], n_calls=30, random_state=42)
        best_nt = NanoTensor((1,), max_order=int(res.x[0]),
                            base_vars=[f'x{i}' for i in range(X.shape[1])])
        best_nt.generate_taylor({f'x{i}': 0 for i in range(X.shape[1])})

        return best_nt

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch prediction for numeric data"""
        return np.array([self.predict({f'x{i}': X[j,i] for i in range(X.shape[1])}).flat[0]
                        for j in range(len(X))])

    def multi_obj_fit(self, sparse_data: List, dense_data: np.ndarray,
                     alpha_exact: float = 0.7):
        """Pareto optimization: combine symbolic (sparse) and numeric (dense)."""
        if dense_data.shape[0] > 0:
            X, y = dense_data[:, :-1], dense_data[:, -1]
            nt_dense = self.symbolic_regression(X, y)
            trainer_dense = HybridTrainer(nt_dense)
            trainer_dense.fit([({f'x{i}': X[j,i] for i in range(X.shape[1])}, y[j])
                              for j in range(len(y))], 'lsq')
        else:
            trainer_dense = self

        trainer_sparse = HybridTrainer(self.nt)
        trainer_sparse.fit(sparse_data, 'symbolic')

        all_keys = set(list(trainer_sparse.fitted_coeffs.keys()) +
                      list(trainer_dense.fitted_coeffs.keys()))

        for k in all_keys:
            val_sparse = trainer_sparse.fitted_coeffs.get(k, 0)
            val_dense = trainer_dense.fitted_coeffs.get(k, 0)
            self.fitted_coeffs[k] = alpha_exact * val_sparse + (1 - alpha_exact) * val_dense

        self._apply_solution(self.fitted_coeffs)

    def torch_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Fit tensor coefficients using PyTorch gradient descent.

        This provides neural network-style optimization for cases where
        symbolic or least-squares methods are insufficient.

        Args:
            X: Input features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Mini-batch size
            verbose: Print training progress

        Returns:
            Dictionary with training history and final coefficients

        Raises:
            ImportError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for torch_fit. "
                "Install with: pip install torch"
            )

        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Convert to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        n_features = X.shape[1]
        n_coeffs = len(self.nt.coeff_vars)

        if n_coeffs == 0:
            """Perform forward operation.

            Args:
            x: Description needed

            Returns:
            Result of the operation

            Example:
            >>> result = obj.forward(...)
            """
            """Perform forward operation.

            Args:
            x: Description needed

            Returns:
            Result of the operation

            Example:
            >>> result = obj.forward(...)
            """
            # Generate Taylor expansion if not already done
            center = {f'x{i}': 0.0 for i in range(n_features)}
            self.nt.generate_taylor(center)
            n_coeffs = len(self.nt.coeff_vars)

        # Create a simple linear model for coefficient learning
        class CoeffModel(nn.Module):
            def __init__(self, n_features, n_coeffs):
                super().__init__()
                # Map from input features to coefficient contributions
                self.linear = nn.Linear(n_features, 1, bias=True)
                # Learnable coefficients
                self.coeffs = nn.Parameter(torch.zeros(n_coeffs))

            def forward(self, x):
                # Simple linear combination
                base = self.linear(x)
                # Add polynomial terms based on coefficients
                poly_terms = torch.zeros_like(base)
                for i, coeff in enumerate(self.coeffs):
                    if i < x.shape[1]:
                        poly_terms += coeff * x[:, i:i+1]
                    elif i < 2 * x.shape[1]:
                        idx = i - x.shape[1]
                        poly_terms += coeff * (x[:, idx:idx+1] ** 2)
                return base + poly_terms

        model = CoeffModel(n_features, n_coeffs)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training history
        history = {'loss': [], 'epochs': epochs}

        # Training loop
        n_samples = X_tensor.shape[0]
        for epoch in range(epochs):
            model.train()

            # Mini-batch training
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]

                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history['loss'].append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

        # Extract fitted coefficients
        model.eval()
        with torch.no_grad():
            learned_coeffs = model.coeffs.numpy()

        # Map back to symbolic coefficients
        for i, coeff_var in enumerate(self.nt.coeff_vars):
            if i < len(learned_coeffs):
                self.fitted_coeffs[str(coeff_var)] = float(learned_coeffs[i])

        # Apply to tensor
        self._apply_solution(self.fitted_coeffs)

        history['final_loss'] = history['loss'][-1] if history['loss'] else 0.0
        history['coefficients'] = self.fitted_coeffs.copy()

        return history


class KnowledgeBase:
    """Symbolic knowledge base using graph and logic programming"""

    def __init__(self):
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
            self._facts = {}

        if KANREN_AVAILABLE:
            self.rules = Relation('policy')
        else:
            self.rules = None

    def add_fact(self, entity: str, prop: str, value: Any):
        """Add fact to KB"""
        if self.graph is not None:
            self.graph.add_edge(entity, prop, value=float(value))
        else:
            key = (entity, prop)
            self._facts[key] = value

        if KANREN_AVAILABLE and self.rules is not None:
            facts(self.rules, (entity, prop, value))

    def query(self, entity: str, prop: str = None) -> List[Any]:
        """Query KB for entity/property"""
        if KANREN_AVAILABLE and self.rules is not None and prop:
            q = kvar()
            results = run(5, q, self.rules(entity, prop, q))
            return [float(r) for r in results]
        elif self.graph is not None:
            if prop:
                if self.graph.has_edge(entity, prop):
                    return [self.graph.edges[entity, prop].get('value')]
            else:
                if self.graph.has_node(entity):
                    return list(self.graph.out_edges(entity, data=True))
        else:
            # Fallback to simple dict
            if prop:
                key = (entity, prop)
                if key in self._facts:
                    return [self._facts[key]]
            else:
                return [(e, p, v) for (e, p), v in self._facts.items() if e == entity]
        return []
