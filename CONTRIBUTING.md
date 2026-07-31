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
   git remote add upstream https://github.com/uttampaliwal/stomar.git
   ```
4. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (installs and manages Python 3.13 automatically)
- Node.js 22.22+ (see `web/.nvmrc`)
- Git

### Installation

```bash
# Sync Python environment (creates .venv with pinned Python, installs all deps from uv.lock)
uv sync

# Install frontend dependencies (web/package-lock.json)
cd web && npm ci && cd ..

# Run lint/tests (no manual activation needed)
uv run ruff check .
uv run pytest tests/ -v --tb=short
```

`pyproject.toml` is the single source of truth for Python dependencies
(`requirements.txt` no longer exists). Pin tooling with uv commands, e.g.
`uv add pandas` / `uv add --dev pytest`, and commit the updated `uv.lock`.

### Environment Variables

All configuration is read from `STOMAR_*` environment variables (see `src/core/settings.py` and the
`.env.example` file). No API keys are required for core functionality:
- **yfinance** for market data (free, no API key needed)
- **FinBERT** for sentiment analysis (auto-downloaded on first use)

Common overrides:
```bash
export STOMAR_ENV=dev               # dev or production
export STOMAR_API_KEY=your_key_here # optional auth for the API
export STOMAR_CORS_ORIGINS=http://localhost:5173
```

## Code Style

### Linting

We use **ruff** for linting. Run before committing:

```bash
uv run ruff check . --output-format=concise
```

Auto-fix common issues:
```bash
uv run ruff check . --fix
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
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_model.py -v

# Run specific test
uv run pytest tests/test_model.py::TestModel::test_specific -v
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
2. **Run linting**: `uv run ruff check .`
3. **Run full test suite**: `uv run pytest tests/ -v`
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

Report security concerns privately via the repository's
[Security tab](https://github.com/uttampaliwal/stomar/security) (private vulnerability reporting).

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
├── start_dev.sh / start_dev.bat  # One-click dev startup (uv sync + npm ci + both servers)
├── run_daily.py                  # Autonomous daily loop
├── run_pipeline.py               # Retraining pipeline
├── auto_pipeline.py              # Startup automation
├── schedule_pipeline.py          # Windows Task Scheduler
├── verify_system.py              # System verification
├── pyproject.toml                # Python deps + ruff/pytest config (single source of truth)
├── uv.lock                       # Locked dependency graph (commit it)
├── .python-version               # Pinned Python 3.13
├── api/                          # FastAPI backend
│   ├── main.py                   # App + CORS + response caching
│   ├── utils.py                  # Parallel fetch utility
│   └── routers/                  # 24 API routers
├── web/                          # React frontend (React 19 + Vite + TypeScript)
│   ├── src/
│   │   ├── pages/                # 21 page components
│   │   ├── components/           # Sidebar, ThemeProvider, UI components
│   │   ├── hooks/                # useApi, usePostApi, useDebouncedValue
│   │   └── lib/                  # Utilities + typed API shapes (api-types.ts)
│   ├── .nvmrc                    # Node 22
│   └── vite.config.ts            # Proxy /api → :8000
├── src/                          # Python ML/trading logic
│   ├── core/                     # Settings, constants, logging, pipeline, backfill
│   ├── data/                     # Data fetch, features, feature store
│   ├── models/                   # 5 model architectures, trainer, ensemble, meta-controller
│   ├── signals/                  # 16 signal modules (sentiment, flow, regime, risk, ...)
│   └── trading/                  # Engine, paper trader, portfolio, backtester, ledger, optimizer
├── tests/                        # 750 tests (pytest)
├── data/                         # Runtime data cache (gitignored)
├── models/                       # Trained weights (gitignored)
└── .github/workflows/ci.yml      # CI: ruff + pytest + web lint/build
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

- Open a discussion on GitHub
- Check existing issues and documentation

Thank you for contributing to StoMar!
