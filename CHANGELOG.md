# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-02-03

### Added
- Initial public release
- Support for multiple model architectures: Random Forest, CNN, LSTM, Bidirectional LSTM
- Comprehensive hyperparameter grid search framework using Sacred
- Evaluation metrics: accuracy, precision, recall, ROC AUC
- Data loading and preprocessing pipeline
- Visualization utilities for learning curves and ROC curves
- Jupyter notebooks for data exploration and analysis
- Complete documentation and setup instructions
- Contributing guidelines and code of conduct
- MIT License
- Type hints and docstrings for main modules

### Fixed
- Fixed typo in models.py: "Bidrectional" → "Bidirectional"
- Removed hardcoded user paths from config.py
- Added missing ConfigSpace dependency to requirements.txt

### Changed
- Refactored config.py to use environment variables for data paths
- Enhanced package structure with proper `__init__.py`
- Improved documentation and README

### Security
- Added .gitignore to prevent accidental data uploads
- Configured git to exclude sensitive paths

## [Unreleased]

### Planned
- Unit tests for core modules
- GitHub Actions CI/CD pipeline
- Code formatting (black, isort)
- Linting (flake8, mypy)
- Extended API documentation
- Example scripts and tutorials
- Support for additional model architectures
- Improved data preprocessing pipeline
