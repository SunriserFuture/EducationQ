"""
Basic tests for EducationQ Framework
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported"""
    try:
        # Test basic imports
        import pandas as pd
        import yaml
        import tiktoken
        from openai import OpenAI
        from datasets import load_dataset
        
        assert True, "All required packages can be imported"
    except ImportError as e:
        pytest.fail(f"Failed to import required package: {e}")

def test_config_template_exists():
    """Test that config template exists"""
    config_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'src', 
        'data', 
        'input', 
        'config_template.yaml'
    )
    assert os.path.exists(config_path), f"Config template not found at {config_path}"

def test_main_script_exists():
    """Test that main script exists"""
    script_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'src', 
        'run', 
        'main.py'
    )
    assert os.path.exists(script_path), f"Main script not found at {script_path}"

if __name__ == "__main__":
    pytest.main([__file__]) 