#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run all tests for the Celeris library.
"""

import sys
import os
import importlib.util

# Add the parent directory to the path so we can import celeris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_all_tests():
    """Run all tests for the Celeris library."""
    print("Running all tests for the Celeris library...\n")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of test modules to run
    test_modules = [
        "test_basic.py",
        "test_warp_specialization.py",
        "test_register_usage.py",
        "test_overlapping_operations.py",
        "test_block_scheduling.py",
        "test_jit_compilation.py",
        "test_unaligned_block_sizes.py",
        "test_tensor_cores.py",
    ]
    
    # Run each test module
    for test_module in test_modules:
        test_path = os.path.join(script_dir, test_module)
        if os.path.exists(test_path):
            print(f"\n{'='*80}")
            print(f"Running {test_module}...")
            print(f"{'='*80}")
            
            try:
                # Import the module
                module_name = os.path.splitext(test_module)[0]
                module = import_module_from_file(module_name, test_path)
                
                # Find and run the main test function
                if hasattr(module, "main"):
                    module.main()
                elif hasattr(module, module_name):
                    getattr(module, module_name)()
                else:
                    print(f"Could not find a test function in {test_module}")
            except Exception as e:
                print(f"Error running {test_module}: {e}")
        else:
            print(f"Test module {test_module} not found at {test_path}")
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    run_all_tests() 