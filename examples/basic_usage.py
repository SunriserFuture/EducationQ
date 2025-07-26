"""
Basic usage example for EducationQ Framework
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Basic usage example"""
    print("EducationQ Framework - Basic Usage Example")
    print("=" * 50)
    
    # Example 1: Run complete pipeline
    print("\n1. Run complete evaluation pipeline:")
    print("   python src/run/main.py")
    
    # Example 2: Use custom config
    print("\n2. Use custom configuration:")
    print("   python src/run/main.py --config ../data/input/my_config.yaml")
    
    # Example 3: Resume from pretest results
    print("\n3. Resume from pretest results:")
    print("   python src/run/main.py --mode load_pretest --input pretest_results.json")
    
    # Example 4: Run specific evaluation
    print("\n4. Run specific evaluation:")
    print("   python src/run/main.py --mode evaluation --posttest posttest.json --csv evaluation_tasks.csv --eval-type comprehensive")
    
    print("\nFor more details, see README.md")

if __name__ == "__main__":
    main() 