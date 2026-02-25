# Dukkit - Defect μscopy toolkit

# Install in editable mode (auto-creates .venv if needed)
install:
	uv pip install -e .

# Install with dev dependencies (prospector, ruff)
install-dev:
	uv pip install -e .[dev]

# Install cpufit from local wheel (platform-specific)
install-cpufit:
	uv run python install_gpufit_wheels.py cpufit

# Install gpufit from local wheel (Windows only)
install-gpufit:
	uv run python install_gpufit_wheels.py gpufit

# Install both cpufit and gpufit (where available)
install-fit-backends:
	uv run python install_gpufit_wheels.py both

# Build wheel and sdist
build:
	rm -rf dist
	uv build

# Quick smoke test - can we import it and check version?
check:
	uv run python -c "import dukit; print(f'dukit {dukit.__version__}')"

# Run all tests
test: install-dev
	uv run pytest tests/ -v

# Run tests with coverage report
test-cov: install-dev
	uv run pytest tests/ -v --cov=src/dukit --cov-report=html --cov-report=term

# Run only field reconstruction tests
test-field: install-dev
	uv run pytest tests/test_field_reconstruction.py -v

# Run prospector - always succeeds, check prospector.log for issues
lint: install-dev
	uv run prospector --profile dukit.prospector.yaml -o grouped:prospector.log || true

# Run prospector - fails if issues found (for CI)
lint-strict: install-dev
	uv run prospector --profile dukit.prospector.yaml -o grouped:prospector.log

# Build documentation
docs:
	uv run pdoc3 --output-dir docs --html --template-dir ./docs/ --force ./src/dukit

# Format code with ruff
format:
	uv run ruff format src/dukit

# Check code formatting with ruff (no changes)
check-format:
	uv run ruff check --select E,W,F,I src/dukit

# Auto-fix issues with ruff
fix:
	uv run ruff check --fix src/dukit
