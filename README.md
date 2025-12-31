# Algorithm-Tumbler
mash algorithms.

## Integration assessment (Symbo components)
- **SymboLLM / SymboLLM_Core**: Transformer-based reasoning engine plus knowledge store that can learn from problem/solution pairs. Integrating it would give Algorithm-Tumbler a trainable generative layer for pattern mining and retrieval-augmented reasoning. It needs light adapters to feed tumble outputs into `LLMTask` prompts and to persist checkpoints in `data/symbo_llm/`.
- **Nano_Tensor**: Symbolic tensor math (polynomial solving, Taylor expansion, hybrid symbolic-numeric training) without SymPy. Useful for exact/approximate algebra inside mash steps. Likely needs slim wrappers so existing algorithm modules call its solver APIs.
- **Reasoning_Tensor**: Nano_Tensor derivative for word-problem state tracking (entity timelines, constraint accumulation, sign inference). Helpful when handling narrative problems. Requires mapping parsed problem states from the current pipeline into its entity/constraint structures.
- **Overall value**: Incorporating these components would elevate mash outputs from rule-based recombination to learnable, auditable reasoning. Integration effort centers on wiring data flow (problems/solutions -> SymboLLM, symbolic expressions -> tensor engines) rather than core changes to the components themselves.
