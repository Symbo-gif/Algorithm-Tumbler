# Algorithm-Tumbler
mash algorithms 

## Integration assessment (Symbo components)
- **SymboLLM / SymboLLM_Core**: Provides the transformer-based reasoning engine and knowledge store that can learn from problem/solution pairs. Integrating it would give Algorithm-Tumbler a trainable generative layer for pattern mining and retrieval-augmented reasoning. Valuable if we add light adapters to feed tumble outputs into `LLMTask` prompts and to persist checkpoints in `data/symbo_llm/...`.
- **Nano_Tensor**: Supplies symbolic tensor math (polynomial solving, Taylor expansion, hybrid symbolic-numeric training) without SymPy. Useful for exact/approximate algebra inside mash steps; likely needs slim wrappers so existing algorithm modules call its solver APIs.
- **Reasoning_Tensor**: Specializes Nano_Tensor for word-problem state tracking (entity timelines, constraint accumulation, sign inference). Worth adding when handling narrative problems; requires mapping parsed problem states from current pipeline into its entity/constraint structures.
- **Overall value**: Incorporating these components would elevate mash outputs from rule-based recombination to learnable, auditable reasoning. Integration effort centers on wiring data flow (problems/solutions -> SymboLLM, symbolic expressions -> tensor engines) rather than core changes to the components themselves.
