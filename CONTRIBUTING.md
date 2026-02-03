# Contributing to Drowsiness Detection

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

This project is committed to providing a welcoming and inspiring community for all. Please read and follow our Code of Conduct.

## How to Contribute

### Reporting Bugs

Before creating bug reports, check the issue list as you might find out that you don't need to create one. When you are creating a bug report, include as many details as possible:

- **Use a clear, descriptive title**
- **Describe the exact steps which reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed**
- **Explain which behavior you expected to see instead and why**
- **Include code snippets or error messages**
- **Python version, OS, and other relevant environment info**

### Suggesting Enhancements

When creating enhancement suggestions, include:

- **Use a clear, descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and expected behavior**
- **Explain the motivation for this enhancement**

### Pull Requests

- Follow the Python style guide (PEP 8)
- Include appropriate test cases
- Update documentation as needed
- Use clear, descriptive commit messages
- Reference related issues in PR descriptions

### Development Process

1. **Fork the repository** to your GitHub account
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/drowsiness-detection.git
   cd drowsiness-detection
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```
4. **Make your changes** and test thoroughly
5. **Format code** with Black:
   ```bash
   black drowsiness_detection/
   ```
6. **Run linting**:
   ```bash
   flake8 drowsiness_detection/
   ```
7. **Run tests**:
   ```bash
   pytest tests/
   ```
8. **Commit with clear messages**:
   ```bash
   git commit -am 'Add some feature'
   ```
9. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```
10. **Create a Pull Request** with a description of changes

## Coding Standards

### Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://github.com/PyCPA/isort) for import sorting
- Aim for [Flake8](https://flake8.pycqa.org/) compliance

### Type Hints

Include type hints for function parameters and return values:

```python
def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 10) -> keras.Model:
    """Train a neural network model.
    
    Args:
        X: Input features array
        y: Target labels
        epochs: Number of training epochs
        
    Returns:
        Trained Keras model
    """
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate classification metrics.
    
    Computes accuracy, precision, recall, and F1-score for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels or probabilities
        
    Returns:
        Dictionary containing metrics:
            - accuracy: Overall accuracy
            - precision: Precision score
            - recall: Recall score
            - f1: F1 score
            
    Raises:
        ValueError: If arrays have different lengths
    """
```

### Testing

- Write tests for new functionality in `tests/`
- Use pytest framework
- Aim for >80% code coverage
- Run tests before submitting PR:
  ```bash
  pytest --cov=drowsiness_detection tests/
  ```

## Questions?

Feel free to open an issue for any questions or discussions about the project.

---

**Thank you for contributing to making this project better!**
