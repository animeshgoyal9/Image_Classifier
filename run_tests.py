#!/usr/bin/env python3
"""Test runner script for DocShield."""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run the test suite."""
    print("Running DocShield tests...")
    print("=" * 50)
    
    # Run pytest with coverage (if available)
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short"
    ]
    
    # Add coverage if pytest-cov is available
    try:
        import pytest_cov
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    except ImportError:
        print("Note: pytest-cov not available, running without coverage")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        return e.returncode


def run_specific_tests(test_pattern=None):
    """Run specific tests based on pattern."""
    print(f"Running tests matching pattern: {test_pattern}")
    print("=" * 50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short"
    ]
    
    if test_pattern:
        cmd.append(f"tests/{test_pattern}")
    else:
        cmd.append("tests/")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ Tests completed!")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        return e.returncode


def run_unit_tests():
    """Run only unit tests."""
    print("Running unit tests...")
    print("=" * 50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-m", "unit"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ Unit tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some unit tests failed!")
        return e.returncode


def run_integration_tests():
    """Run only integration tests."""
    print("Running integration tests...")
    print("=" * 50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-m", "integration"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ Integration tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some integration tests failed!")
        return e.returncode


def main():
    """Main function."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "unit":
            return run_unit_tests()
        elif command == "integration":
            return run_integration_tests()
        elif command == "specific":
            pattern = sys.argv[2] if len(sys.argv) > 2 else None
            return run_specific_tests(pattern)
        elif command == "help":
            print("Usage:")
            print("  python run_tests.py              # Run all tests with coverage")
            print("  python run_tests.py unit         # Run only unit tests")
            print("  python run_tests.py integration  # Run only integration tests")
            print("  python run_tests.py specific     # Run specific test pattern")
            print("  python run_tests.py help         # Show this help")
            return 0
        else:
            print(f"Unknown command: {command}")
            print("Use 'python run_tests.py help' for usage information")
            return 1
    else:
        return run_tests()


if __name__ == "__main__":
    sys.exit(main())
