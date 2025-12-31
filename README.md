# Algorithm-Tumbler

Algorithm-Tumbler is a research sandbox for composing, decomposing, and mining program structures while pairing them with symbolic and LLM-backed reasoning. It mixes a lightweight DSL, interpreters, decomposition tools, and Symbo tensor/LLM engines to explore algorithm search and verification.

## What it does
- Build and run programs expressed in a small DSL (`algorithms/core`).
- Analyze code with slicing, call graphing, and summarization utilities (`algorithms/analysis`).
- Decompose loops and strategies for program restructuring (`algorithms/decomposer`).
- Bridge to Symbo components for symbolic math and reasoning (`symbo_llm.py`, `nano_tensor.py`, `reasoning_tensor.py`, `algorithms/symbo_bridge.py`).
- Provide CLI helpers and notebooks for mash experiments (`algorithms/ui/cli.py`, `examples/advanced_tumbling.ipynb`).

## Repository layout
- `algorithms/` – main package
  - `core/` – DSL primitives, types, interpreter, and program model
  - `analysis/` – program slicing, call graphs, semantic summarization
  - `decomposer/` – loop extraction, hierarchical slicing, strategy utilities
  - `evolution/`, `mining/`, `io/`, `ui/` – search, mining, serialization, CLI helpers
  - `symbo_bridge.py` – glue code to Symbo reasoning backends
- `symbo_llm.py` / `symbo_llm_core.py` – Symbo LLM adapter and transformer core
- `nano_tensor.py` / `reasoning_tensor.py` – symbolic tensor engines (math and narrative reasoning)
- `examples/advanced_tumbling.ipynb` – notebook walkthrough
- `tests/` – lightweight smoke tests for core modules and bridges
- `docs/` – PDF overview and security audit notes

## Running tests
Install test dependencies (e.g., `pip install pytest`) and run:
```bash
pytest
```

## License
Algorithm-Tumbler is licensed under the Apache License 2.0. See the [`LICENSE`](LICENSE) file for details.
