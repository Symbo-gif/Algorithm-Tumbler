#!/usr/bin/env python3
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
Self-Tumbling Script: Process all Python files in the repository through the
tumbler system to extract primitives and add them to the component library.
"""

import os
import sys
from pathlib import Path
from typing import List

# Add the repository root to the path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from algorithms.library.manager import ComponentLibrary
from algorithms.decomposer.function_extractor import FunctionLevelDecomposer


def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files in the repository, excluding tests and common directories."""
    python_files = []
    exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', 'env', '.venv'}
    
    for path in root_dir.rglob('*.py'):
        # Skip if any parent directory is in exclude list
        if any(excluded in path.parts for excluded in exclude_dirs):
            continue
        # Skip __init__.py files (they're usually just imports)
        if path.name == '__init__.py':
            continue
        python_files.append(path)
    
    return sorted(python_files)


def tumble_file(file_path: Path, repo_root: Path) -> List:
    """Process a single Python file through the tumbler to extract components."""
    try:
        relative_path = file_path.relative_to(repo_root)
        print(f"  Tumbling: {relative_path}")
    except ValueError:
        print(f"  Tumbling: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        print(f"    Error reading file: {e}")
        return []
    
    # Use the function-level decomposer
    decomposer = FunctionLevelDecomposer()
    components = decomposer.decompose(code)
    
    print(f"    Extracted {len(components)} components")
    return components


def main():
    """Main tumbler process."""
    print("=" * 70)
    print("Algorithm Tumbler: Self-Tumbling Process")
    print("=" * 70)
    print()
    
    repo_root = Path(__file__).parent
    
    # Start with the seed library
    print("1. Loading seed library...")
    library = ComponentLibrary.seed_default()
    print(f"   Initial library has {len(library.components)} components")
    print()
    
    # Find all Python files
    print("2. Finding Python files in repository...")
    python_files = find_python_files(repo_root)
    print(f"   Found {len(python_files)} Python files to process")
    print()
    
    # Process each file
    print("3. Processing files through tumbler system...")
    total_extracted = 0
    
    for py_file in python_files:
        components = tumble_file(py_file, repo_root)
        
        # Add components to library
        for comp in components:
            try:
                # Generate a unique ID based on full relative path and function name
                # This helps avoid collisions between files with the same name
                relative_path = py_file.relative_to(repo_root)
                path_str = str(relative_path.with_suffix('')).replace('/', '_').replace('\\', '_')
                comp_id = f"{path_str}_{comp.pattern.name}"
                library.add(comp, comp_id)
                total_extracted += 1
            except Exception as e:
                print(f"    Warning: Could not add component: {e}")
    
    print()
    print(f"   Total components extracted and added: {total_extracted}")
    print()
    
    # Save the extended library
    print("4. Saving extended library...")
    output_path = repo_root / "library_self_tumbled.json"
    library.save(str(output_path))
    print(f"   Saved to: {output_path}")
    print()
    
    # Summary
    print("=" * 70)
    print("Self-Tumbling Complete!")
    print("=" * 70)
    print(f"Total components in library: {len(library.components)}")
    print(f"  - Seed components: {len(ComponentLibrary.seed_default().components)}")
    print(f"  - Extracted components: {total_extracted}")
    print()
    print(f"Library saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
