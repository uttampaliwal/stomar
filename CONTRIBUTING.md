# Contributing to StoMar

Thank you for your interest in contributing to StoMar! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Code of Conduct](#code-of-conduct)

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/stomar.git
   cd stomar
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/anomalyco/stomar.git
   ```
4. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- pip
- Git

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install ruff pytest pytest-cov
```

### Environment Variables

No API keys are required for core functionality. The project uses:
- **yfinance** for market data (free, no API key needed)
- **FinBERT** for sentiment analysis (auto-downloaded on first use)

Optional environment variables:
```bash
# For enhanced data sources (optional)
export NSE_ARCHIVE_API_KEY=your_key_here
```

## Code Style

### Linting

We use **ruff** for linting. Run before committing:

```bash
python -m ruff check src/ tests/ --output-format=concise
```

Auto-fix common issues:
```bash
python -m ruff check src/ tests/ --fix
```

### Formatting Guidelines

- **Line length**: 88 characters (ruff default)
- **Quotes**: Double quotes for strings
- **Imports**: Sorted by ruff (isort-compatible)
- **Type hints**: Use Python 3.10+ syntax (`list`, `dict`, `|` instead of `List`, `Dict`, `Optional`)

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: Prefix with `_` (e.g., `_internal_function`)

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_model.py -v

# Run specific test
python -m pytest tests/test_model.py::TestModel::test_specific -v
```

### Writing Tests

- Place tests in `tests/` directory
- File naming: `test_<module>.py`
- Class naming: `Test<Feature>`
- Method naming: `test_<behavior>`
- Use `pytest` fixtures for shared setup
- Mock external dependencies (API calls, file I/O)

Example:
```python
import pytest
from src.your_module import your_function

class TestYourFunction:
    def test_returns_expected_type(self):
        result = your_function("input")
        assert isinstance(result, dict)

    def test_handles_edge_case(self):
        result = your_function("")
        assert result == {}

    def test_raises_on_invalid_input(self):
        with pytest.raises(ValueError):
            your_function(None)
```

### Test Coverage

Aim for:
- **80%+ line coverage** on new code
- **All public functions** have tests
- **Edge cases** and **error paths** covered
- **No network calls** in tests (mock external services)

## Pull Request Process

### Before Submitting

1. **Update tests** for any new/changed functionality
2. **Run linting**: `python -m ruff check src/ tests/`
3. **Run full test suite**: `python -m pytest tests/ -v`
4. **Update documentation** if adding new features
5. **Keep commits atomic**: One logical change per commit

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added (if applicable)
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Documentation updated (if needed)
- [ ] No new warnings introduced
```

### Review Process

1. PR will be reviewed by maintainers
2. Address review feedback
3. Once approved, PR will be merged

## Reporting Issues

### Bug Reports

Include:
- **OS and Python version**
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Error messages/tracebacks**
- **Minimal code example** if applicable

### Security Vulnerabilities

**Do not open public issues for security vulnerabilities.**

Email security concerns to: [SECURITY_EMAIL]

## Feature Requests

We welcome feature requests! Please include:
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Implementation ideas**: If you have thoughts on how to build it

## Areas for Contribution

Looking for help with:

- [ ] Additional data sources (NSE, BSE APIs)
- [ ] More technical indicators
- [ ] Portfolio optimization strategies
- [ ] Backtesting improvements
- [ ] Documentation and examples
- [ ] Performance optimization
- [ ] Unit and integration tests
- [ ] Bug fixes (check issues labeled "good first issue")

## Project Structure

```
stomar/
├── src/                    # Main source code
│   ├── __init__.py
│   ├── data_fetcher.py     # Data acquisition
│   ├── features.py         # Feature engineering
│   ├── model.py            # ML models
│   ├── ensemble.py         # Ensemble methods
│   ├── backtester.py       # Backtesting engine
│   ├── sentiment.py        # Sentiment analysis
│   ├── risk.py             # Risk metrics
│   └── ...
├── tests/                  # Test suite
├── docs/                   # Documentation
├── data/                   # Data cache (gitignored)
├── models/                 # Trained models (gitignored)
├── app.py                  # Streamlit web app
├── run_pipeline.py         # CLI pipeline
└── requirements.txt        # Dependencies
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

- Open a discussion on GitHub
- Check existing issues and documentation

Thank you for contributing to StoMar!
