# Contributing to Face Enhancement AI

Thank you for your interest in contributing to Face Enhancement AI! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU/CPU)
- Relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- Clear description of the proposed feature
- Use cases and benefits
- Potential implementation approach (if applicable)

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/alexv879/Security-Camera-Face-Image-Enhancement.git
   cd Security-Camera-Face-Image-Enhancement
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Make your changes**
   - Write clear, documented code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

6. **Run tests and checks**
   ```bash
   # Format code
   black src tests
   isort src tests

   # Lint
   flake8 src tests

   # Type check
   mypy src

   # Run tests
   pytest tests/ -v --cov=src/face_enhancement
   ```

7. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes (formatting, etc.)
   - `refactor:` Code refactoring
   - `test:` Adding or updating tests
   - `chore:` Maintenance tasks

8. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function arguments and return values
- Maximum line length: 100 characters
- Use docstrings for all public functions and classes

Example:
```python
def process_image(
    image: np.ndarray,
    upscale: int = 2,
    denoise: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Process and enhance an image.

    Args:
        image: Input image array (BGR format)
        upscale: Upscaling factor (1-4)
        denoise: Whether to apply denoising

    Returns:
        Tuple of (enhanced_image, metadata)
    """
    # Implementation
    pass
```

### Testing

- Write unit tests for new functionality
- Aim for >80% code coverage
- Use pytest fixtures for common test data
- Mock external dependencies (models, file I/O) when appropriate

Example:
```python
import pytest
import numpy as np

@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

def test_enhancement(sample_image):
    result = enhance_image(sample_image)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
```

### Documentation

- Update README.md for new features
- Add docstrings to all public APIs
- Include code examples for complex functionality
- Update configuration documentation if adding new options

### Performance Considerations

- Profile code for performance bottlenecks
- Use GPU acceleration when available
- Implement batch processing for efficiency
- Consider memory usage for large images/videos

## Project Structure

```
Security-Camera-Face-Image-Enhancement/
├── src/face_enhancement/      # Main source code
│   ├── core/                  # Core functionality (detection, pipeline)
│   ├── models/                # Enhancement models
│   ├── preprocessing/         # Image preprocessing
│   ├── api/                   # REST API server
│   └── utils/                 # Utilities
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── config/                    # Configuration files
├── examples/                  # Usage examples
├── docs/                      # Documentation
└── models/                    # Model weights directory
```

## Adding New Models

To add a new enhancement model:

1. Create model class in `src/face_enhancement/models/`
2. Implement standard interface:
   ```python
   class NewEnhancer:
       def __init__(self, config):
           pass

       def enhance(self, image: np.ndarray) -> np.ndarray:
           pass
   ```
3. Update `FaceEnhancer` to support the new model
4. Add configuration options
5. Add tests
6. Update documentation

## Release Process

1. Update version in `setup.py` and `pyproject.toml`
2. Update CHANGELOG.md
3. Create release tag: `git tag v1.x.x`
4. Push tag: `git push origin v1.x.x`
5. Create GitHub release with notes

## Questions?

Feel free to:
- Open an issue for questions
- Join discussions in existing issues
- Reach out to maintainers

Thank you for contributing! 🎉
