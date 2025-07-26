# Contributing to EducationQ Framework

Thank you for your interest in contributing to EducationQ Framework! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment
4. Install development dependencies

```bash
git clone https://github.com/your-username/EducationQ.git
cd EducationQ/EducationQ_Framework
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Development Setup

### Code Formatting

We use `black` for code formatting and `isort` for import sorting:

```bash
# Format code
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/
```

### Linting

We use `flake8` for linting:

```bash
flake8 src/ tests/ examples/
```

### Type Checking

We use `mypy` for type checking:

```bash
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=educationq

# Run specific test file
pytest tests/test_basic.py
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

### Commit Message Format

Use conventional commit messages:

```
feat: add new evaluation method
fix: resolve token counting issue
docs: update README with new examples
test: add tests for student response analysis
```

## Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Use meaningful variable and function names

## Testing Guidelines

- Write unit tests for new functionality
- Aim for at least 80% code coverage
- Use descriptive test names
- Test both success and failure cases

## Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions and classes
- Update examples if API changes
- Keep documentation in sync with code changes

## Issues and Bug Reports

When reporting issues:

1. Use the issue template
2. Provide a clear description of the problem
3. Include steps to reproduce
4. Share relevant error messages
5. Specify your environment (OS, Python version, etc.)

## Questions and Discussion

For questions and general discussion:

- Use GitHub Discussions
- Join our community chat (if available)
- Email: educationq@sunriser.org

## License

By contributing to EducationQ Framework, you agree that your contributions will be licensed under the MIT License. 