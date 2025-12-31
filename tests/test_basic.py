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

# Basic test script
try:
    from algorithms.core import INT, DSLPrimitive, Program
    print("✓ Core imports successful")

    # Test type
    assert INT.name == "int"
    print("✓ Types work")

    # Test node
    node = DSLPrimitive([], INT, [])
    assert node.output_type == INT
    print("✓ Nodes work")

    # Test program
    prog = Program([node], {})
    assert len(prog.nodes) == 1
    print("✓ Programs work")

    print("All basic tests passed!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
