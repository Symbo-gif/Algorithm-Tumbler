# Self-Tumbling System

This document describes the self-tumbling feature of Algorithm Tumbler, which processes Python code files to extract reusable primitives and add them to the component library.

## What is Tumbling?

The tumbling system analyzes Python source code and extracts functions as reusable components (primitives) that can be added to the Algorithm Tumbler library. This allows the system to learn from its own codebase and build a richer library of algorithmic building blocks.

## How to Use

### Using the Standalone Script

Run the self-tumbling script directly:

```bash
python3 tumble_self.py
```

This will:
1. Start with the seed library (4 base components)
2. Scan all Python files in the repository
3. Extract functions as components
4. Save the extended library to `library_self_tumbled.json`

### Using the CLI

Use the `tumble` command from the CLI:

```bash
python3 -m algorithms.ui.cli tumble --directory . --output library_extended.json
```

Options:
- `--directory`: Directory to scan for Python files (default: current directory)
- `--output`: Output file for the extended library (default: `library_self_tumbled.json`)

## Results

The self-tumbling process on this repository extracts approximately:
- **279 function components** from **35 Python files** (including tumble_self.py itself)
- Combined with the 4 seed components for a total of **~250 unique components**
  - Note: Some duplicate component IDs are merged (e.g., functions with common names like `__init__`)

Components are organized by module:
- Core DSL functions
- Analysis functions (call graphs, program slicing)
- Decomposer functions (loop extraction, slicing)
- Evolution functions (mutations, search)
- Library management functions
- Tensor and symbolic reasoning functions

## Output

The generated library file (`library_self_tumbled.json`) contains:
- All seed components (sort, binary search, etc.)
- Extracted function signatures with:
  - Parameter schemas
  - Type information
  - Cost estimates
  - Specifications (from docstrings)

## Example

After tumbling, you can load and query the extended library:

```python
from algorithms.library.manager import ComponentLibrary

# Load the tumbled library
lib = ComponentLibrary.load('library_self_tumbled.json')

# Query by specification
slicing_components = lib.query_by_spec('slice')

# List all components
print(f"Total components: {len(lib.components)}")
```

## Technical Details

The tumbling system uses the `FunctionLevelDecomposer` to:
1. Parse Python files into AST (Abstract Syntax Tree)
2. Walk the AST to find function definitions
3. Extract function metadata (name, parameters, docstring)
4. Create Component objects with:
   - Function primitive wrappers
   - Type information (simplified to INT by default)
   - Cost estimates based on parameter count
   - Specifications from docstrings

## Future Enhancements

Potential improvements to the tumbling system:
- More sophisticated type inference
- Extraction of classes and methods
- Pattern recognition for common idioms
- Semantic clustering of similar components
- Better cost estimation based on complexity analysis
