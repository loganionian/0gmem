# Contributing to 0GMem

Thank you for your interest in contributing to 0GMem! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We're building something together.

## Getting Started

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/loganionian/0gmem.git
   cd 0gmem
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Download required models**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=zerogmem

# Run specific test file
pytest tests/test_integration.py -v
```

### Code Style

We use the following tools for code quality:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/

# Check types
mypy src/zerogmem/
```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/loganionian/0gmem/issues)
2. If not, create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS

### Suggesting Features

1. Check existing issues and discussions
2. Create a new issue with:
   - Clear description of the feature
   - Use case / motivation
   - Potential implementation approach (optional)

### Submitting Pull Requests

1. **Fork the repository** and create a branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Run the test suite** to ensure nothing is broken
6. **Create a pull request** with:
   - Clear description of changes
   - Link to related issue (if any)
   - Screenshots/examples if applicable

### PR Checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Code is formatted (`black --check src/ tests/`)
- [ ] Linting passes (`ruff check src/`)
- [ ] Type hints are valid (`mypy src/zerogmem/`)
- [ ] Documentation is updated (if needed)
- [ ] CHANGELOG.md is updated (for significant changes)

## Project Structure

```
0gmem/
├── src/zerogmem/       # Main package
│   ├── encoder/        # Text encoding
│   ├── memory/         # Memory hierarchy
│   ├── retriever/      # Memory retrieval
│   ├── graph/          # Unified Memory Graph
│   ├── reasoning/      # Advanced reasoning
│   └── evaluation/     # Benchmarking
├── tests/              # Test suite
├── examples/           # Usage examples
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for contributing to 0GMem!
