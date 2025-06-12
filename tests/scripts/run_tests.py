#!/usr/bin/env python3
"""
Test runner script for AngelaMCP.

This script provides various options for running the test suite with
different configurations and filtering options.
"""

import sys
import argparse
import subprocess
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n🚀 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run AngelaMCP tests")
    
    # Test selection options
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run only integration tests"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Run fast tests only (excludes slow and integration)"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all tests including slow ones"
    )
    
    # Coverage options
    parser.add_argument(
        "--no-cov", 
        action="store_true", 
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "--cov-html", 
        action="store_true", 
        help="Generate HTML coverage report"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true", 
        help="Quiet output"
    )
    
    # Specific test options
    parser.add_argument(
        "--test", 
        type=str, 
        help="Run specific test file or test function"
    )
    parser.add_argument(
        "--keyword", "-k", 
        type=str, 
        help="Run tests matching keyword expression"
    )
    
    # Debug options
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Run with debug output"
    )
    parser.add_argument(
        "--pdb", 
        action="store_true", 
        help="Drop into PDB on failures"
    )
    
    # Performance options
    parser.add_argument(
        "--parallel", "-n", 
        type=int, 
        help="Run tests in parallel (number of workers)"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Test selection
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.fast:
        cmd.extend(["-m", "not slow and not integration"])
    elif not args.all:
        # Default: run unit tests and fast integration tests
        cmd.extend(["-m", "not slow"])
    
    # Coverage options
    if not args.no_cov:
        cmd.extend(["--cov=src"])
        if args.cov_html:
            cmd.extend(["--cov-report=html"])
        cmd.extend(["--cov-report=term-missing"])
    
    # Output options
    if args.verbose:
        cmd.append("-v")
    elif args.quiet:
        cmd.append("-q")
    
    # Specific test
    if args.test:
        cmd.append(args.test)
    
    # Keyword filtering
    if args.keyword:
        cmd.extend(["-k", args.keyword])
    
    # Debug options
    if args.debug:
        cmd.extend(["--log-cli-level=DEBUG"])
    if args.pdb:
        cmd.append("--pdb")
    
    # Parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Run the tests
    success = run_command(cmd, "Running tests")
    
    if success:
        print("\n🎉 All tests passed!")
        
        # Show coverage summary if coverage was enabled
        if not args.no_cov:
            print("\n📊 Coverage Summary:")
            subprocess.run(["python", "-m", "coverage", "report", "--show-missing"])
            
            if args.cov_html:
                print("\n📝 HTML coverage report generated in htmlcov/")
                print("Open htmlcov/index.html in your browser to view detailed coverage")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()