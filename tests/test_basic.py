#!/usr/bin/env python3

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
