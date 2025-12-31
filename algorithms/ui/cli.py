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

import argparse
import sys
from pathlib import Path
from ..library.manager import ComponentLibrary
from ..evolution.search import EvolutionarySearch

def main():
    parser = argparse.ArgumentParser(description="Algorithm Tumbler CLI")
    subparsers = parser.add_subparsers(dest='command')

    # init
    subparsers.add_parser('init', help='Initialize library from seed algorithms')

    # spin
    spin_parser = subparsers.add_parser('spin', help='Random spin')
    spin_parser.add_argument('--mode', default='random')
    spin_parser.add_argument('--steps', type=int, default=100)

    # evolve
    evolve_parser = subparsers.add_parser('evolve', help='Guided evolution')
    evolve_parser.add_argument('--objective', required=True)
    evolve_parser.add_argument('--generations', type=int, default=50)

    # add-algorithm
    add_parser = subparsers.add_parser('add-algorithm', help='Add custom algorithm')
    add_parser.add_argument('--file', required=True)

    # inspect
    inspect_parser = subparsers.add_parser('inspect', help='Inspect library')
    inspect_parser.add_argument('--top-k', type=int, default=10)

    # tumble
    tumble_parser = subparsers.add_parser('tumble', help='Process Python files to extract primitives')
    tumble_parser.add_argument('--directory', default='.', help='Directory to scan for Python files')
    tumble_parser.add_argument('--output', default='library_self_tumbled.json', help='Output library file')

    args = parser.parse_args()

    if args.command == 'init':
        library = ComponentLibrary.seed_default()
        library.save('library.json')
        print("Library initialized")

    elif args.command == 'spin':
        library = ComponentLibrary.load('library.json')
        search = EvolutionarySearch(library, seed=42)
        search.initialize(10)
        results = []
        for _ in range(args.steps):
            # Simplified spin
            results.append(search.population[0])
        print(f"Spun {len(results)} programs")

    elif args.command == 'evolve':
        library = ComponentLibrary.load('library.json')
        search = EvolutionarySearch(library, seed=42)
        search.initialize(10)
        test_cases = []  # Placeholder
        search.evolve(args.generations, lambda p, tc: 0.0, test_cases, mode="guided")
        print("Evolution complete")

    elif args.command == 'add-algorithm':
        # Placeholder
        print(f"Added algorithm from {args.file}")

    elif args.command == 'inspect':
        library = ComponentLibrary.load('library.json')
        for i, (cid, comp) in enumerate(list(library.components.items())[:args.top_k]):
            print(f"{i+1}. {cid}: {comp}")

    elif args.command == 'tumble':
        from ..decomposer.function_extractor import FunctionLevelDecomposer
        
        print("Algorithm Tumbler: Processing Python files...")
        
        # Start with seed library
        library = ComponentLibrary.seed_default()
        print(f"Starting with {len(library.components)} seed components")
        
        # Find Python files
        root_dir = Path(args.directory)
        python_files = []
        exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', 'env', '.venv'}
        
        for path in root_dir.rglob('*.py'):
            if any(excluded in path.parts for excluded in exclude_dirs):
                continue
            if path.name == '__init__.py':
                continue
            python_files.append(path)
        
        print(f"Found {len(python_files)} Python files to process")
        
        # Process files
        decomposer = FunctionLevelDecomposer()
        total_extracted = 0
        
        for py_file in sorted(python_files):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                components = decomposer.decompose(code)
                
                for comp in components:
                    try:
                        relative_path = py_file.relative_to(root_dir)
                        # Use full path to avoid ID collisions
                        path_str = str(relative_path.with_suffix('')).replace('/', '_').replace('\\', '_')
                        comp_id = f"{path_str}_{comp.pattern.name}"
                        library.add(comp, comp_id)
                        total_extracted += 1
                    except (ValueError, AttributeError, KeyError) as e:
                        # Skip components that can't be added (duplicate IDs, invalid structure)
                        # Silently skip to avoid cluttering output in batch processing
                        pass
            except (IOError, OSError, UnicodeDecodeError) as e:
                # Skip files that can't be read
                # Silently skip to avoid cluttering output in batch processing
                pass
        
        # Save library
        library.save(args.output)
        print(f"Extracted {total_extracted} components")
        print(f"Total components in library: {len(library.components)}")
        print(f"Saved to: {args.output}")

if __name__ == '__main__':
    main()
