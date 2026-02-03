# Implementation Guide: Code Quality Improvements

This document provides step-by-step instructions to implement the high-priority and medium-priority code quality improvements identified in `CODE_QUALITY_ANALYSIS.md`.

## Phase 1: Critical Fixes (2-3 hours)

These fixes address bugs and major code issues that should be fixed first.

### Fix 1.1: Remove Duplicate Function Definitions in `models.py`

**File:** `drowsiness_detection/models.py`  
**Lines to delete:** 276-325 (entire duplicated code block)

**Current state:** The file has these duplicate definitions:
- `build_lstm_model()` defined at line 163 AND line 276
- `build_bi_lstm_model()` defined at line 204 AND lines after 276

**Action:**
1. Open `drowsiness_detection/models.py`
2. Scroll to line 276
3. Delete from line 276 to line 325 (everything after the `build_bi_lstm_model()` function that ends around line 272)
4. The first definitions (lines 163-245) are properly documented and should be kept
5. Run `pytest tests/test_models.py` to verify tests still pass

**Verification command:**
```bash
cd /home/tim/coding/drowsiness-detection
python -c "from drowsiness_detection.models import build_lstm_model, build_bi_lstm_model; print('Import successful')"
```

---

### Fix 1.2: Fix Undefined Variable `KSS_THRESHOLD` in `data.py`

**File:** `drowsiness_detection/data.py`  
**Line:** 151 in function `get_train_test_splits()`

**Current code:**
```python
y_train, y_test = binarize(y_train, KSS_THRESHOLD), binarize(y_test, KSS_THRESHOLD)
```

**Problem:** `KSS_THRESHOLD` is not defined in this function's scope

**Action:**
1. Create a new file `drowsiness_detection/constants.py` with this content:

```python
"""Global constants and thresholds for drowsiness detection."""

# KSS Thresholds for label discretization
KSS_THRESHOLD_BINARY = 7

# KSS Bins for multi-class classification
KSS_BINS_3CLASS = [6, 8, 10]
KSS_BINS_5CLASS = [3, 5, 7, 9, 10]
KSS_BINS_9CLASS = list(range(1, 10))

# Feature column indices (refers to specific features from raw data)
# These correspond to eye-gaze and eye-closure features
DEFAULT_FEATURE_INDICES = (5, 8, 9, 14, 15, 16, 19)

# File size limits
MAX_CHUNK_SIZE_MB = 100
```

2. In `drowsiness_detection/data.py`, add import at the top:
```python
from drowsiness_detection.constants import KSS_THRESHOLD_BINARY
```

3. In `get_train_test_splits()` function around line 151, replace:
```python
y_train, y_test = binarize(y_train, KSS_THRESHOLD), binarize(y_test, KSS_THRESHOLD)
```
with:
```python
y_train, y_test = binarize(y_train, KSS_THRESHOLD_BINARY), binarize(y_test, KSS_THRESHOLD_BINARY)
```

4. Update `preprocess_feature_data()` function (around line 235) to use the constant:
```python
# Old
if num_targets == 2:
    KSS_THRESHOLD = 7
    targets = binarize(targets, threshold=KSS_THRESHOLD)

# New
if num_targets == 2:
    targets = binarize(targets, threshold=KSS_THRESHOLD_BINARY)
```

5. Add the import to the imports section at top of file.

**Verification:**
```bash
python -c "from drowsiness_detection.data import get_train_test_splits; print('Import successful')"
```

---

### Fix 1.3: Remove Duplicate Label Discretization Logic

**File:** `drowsiness_detection/data.py`  
**Location:** In `preprocess_feature_data()` function (lines ~234-252)

**Current state:** The function contains inline discretization logic that duplicates the `discretize_labels_by_threshold()` function (defined at line 263)

**Action:**
1. In the `preprocess_feature_data()` function, replace the entire if-elif block (lines 234-252):

**Before:**
```python
    # col -3 is targets, -2 is sess type and -1 is subject id
    feature_data = np.nan_to_num(feature_data)
    targets = feature_data[:, -3]
    if num_targets == 2:
        targets = binarize(targets, threshold=KSS_THRESHOLD_BINARY)
    elif num_targets == 3:
        ALERT = 6
        NEUTRAL = 8
        SLEEPY = 10
        targets = np.digitize(targets, bins=[ALERT, NEUTRAL, SLEEPY])
    elif num_targets == 5:
        first = 3
        second = 5
        third = 7
        forth = 9
        fifth = 10
        targets = np.digitize(targets, bins=[first, second, third, forth, fifth])
    elif num_targets == 9:
        targets = np.digitize(targets, bins=range(1, 10))
    else:
        raise ValueError(f"num targets {num_targets} not supported.")
```

**After:**
```python
    # col -3 is targets, -2 is sess type and -1 is subject id
    feature_data = np.nan_to_num(feature_data)
    targets = feature_data[:, -3]
    targets = discretize_labels_by_threshold(targets, num_targets=num_targets)
```

2. Verify `discretize_labels_by_threshold()` is defined earlier in the file

---

### Fix 1.4: Remove Unused Import

**File:** `drowsiness_detection/visualize.py`  
**Line:** 1

**Action:**
1. Remove this line:
```python
from copy import deepcopy
```

**Verification:**
```bash
python -c "from drowsiness_detection.visualize import *; print('Import successful')"
```

---

### Fix 1.5: Remove Typo in Function Name

**File:** `drowsiness_detection/helpers.py`  
**Function name at line 128:** `create_emtpy_array_of_max_size` → `create_empty_array_of_max_size`

**Action:**
1. Find all occurrences of `create_emtpy_array_of_max_size`:
```bash
grep -n "create_emtpy_array_of_max_size" drowsiness_detection/helpers.py
```

2. Rename the function definition (line 128):
```python
# Before
def create_emtpy_array_of_max_size(max_size_mb: int, n_cols: int) -> np.ndarray:

# After
def create_empty_array_of_max_size(max_size_mb: int, n_cols: int) -> np.ndarray:
```

3. Find and update all usages. Search for it in `data.py`:
```bash
grep -n "create_emtpy" drowsiness_detection/data.py
```

Update all occurrences to `create_empty_array_of_max_size`.

---

## Phase 2: Configuration and Logging (3-4 hours)

These changes improve code maintainability through better configuration and logging.

### Fix 2.1: Create Logging Configuration Module

**New file:** `drowsiness_detection/logging_config.py`

```python
"""Logging configuration for drowsiness detection."""
import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    name: str = "drowsiness_detection"
) -> logging.Logger:
    """Configure logging for the package.
    
    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
        log_file: Optional path to log file
        name: Logger name
        
    Returns:
        Configured logger instance
        
    Example:
        >>> from drowsiness_detection.logging_config import setup_logging
        >>> logger = setup_logging(level=logging.DEBUG, log_file=Path("app.log"))
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Module-level logger for convenience
logger = setup_logging()
```

### Fix 2.2: Replace Print Statements with Logging

This is a larger task affecting multiple files. Start with the most critical ones:

#### 2.2a: `drowsiness_detection/metrics.py`

Add at the top:
```python
import logging
logger = logging.getLogger(__name__)
```

Replace function `print_metric_results()`:

**Before:**
```python
def print_metric_results(results: tuple) -> None:
    """Print formatted classification metric results.
    
    Args:
        results: Tuple of (mean, std) pairs for each metric from
                 calc_mean_and_std_of_classification_metrics()
    """
    results = [(round(x, 2), round(y, 3)) for x, y in results]
    (mean_acc, std_acc), (mean_recall, std_recall), (mean_precision, std_precision), (
        mean_auc, std_auc) = results
    print(rf"Mean Accuracy = {mean_acc} ± {std_acc}")
    print(rf"Mean Precision = {mean_precision} ± {std_precision}")
    print(rf"Mean Recall = {mean_recall} ± {std_recall}")
    print(rf"Mean ROC AUC = {mean_auc} ± {std_auc}")
```

**After:**
```python
def print_metric_results(results: tuple, use_logger: bool = True) -> None:
    """Print formatted classification metric results.
    
    Args:
        results: Tuple of (mean, std) pairs for each metric from
                 calc_mean_and_std_of_classification_metrics()
        use_logger: If True, use logging; if False, use print (for backward compatibility)
    """
    results = [(round(x, 2), round(y, 3)) for x, y in results]
    (mean_acc, std_acc), (mean_recall, std_recall), (mean_precision, std_precision), (
        mean_auc, std_auc) = results
    
    message_lines = [
        f"Mean Accuracy = {mean_acc} ± {std_acc}",
        f"Mean Precision = {mean_precision} ± {std_precision}",
        f"Mean Recall = {mean_recall} ± {std_recall}",
        f"Mean ROC AUC = {mean_auc} ± {std_auc}"
    ]
    
    if use_logger:
        for line in message_lines:
            logger.info(line)
    else:
        for line in message_lines:
            print(line)
```

#### 2.2b: `drowsiness_detection/models.py`

Add at the top:
```python
import logging
logger = logging.getLogger(__name__)
```

Replace all `print(model.summary())` calls with `logger.debug(model.summary())`:

**Examples:**
- Line 99: `print(model.summary())` → `logger.debug("Model summary:\n" + model.summary())`
- Line 159: same replacement
- etc.

And fix the warning message:

**Before:**
```python
else:
    print(f"Activation was set to {activation} which is not supported.")
```

**After:**
```python
else:
    logger.warning(f"Activation '{activation}' is not supported, using default.")
```

---

### Fix 2.3: Create Constants Module (if not done in Fix 1.2)

Ensure `drowsiness_detection/constants.py` exists with all necessary constants properly organized.

---

## Phase 3: Validation and Error Handling (2-3 hours)

These changes improve error messages and input validation.

### Fix 3.1: Add Input Validation to Key Functions

#### 3.1a: `drowsiness_detection/data.py` - `get_train_test_splits()`

**Before:**
```python
def get_train_test_splits(directory: Path = config.PATHS.TRAIN_TEST_SPLIT) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test data from split files."""
    for file in directory.iterdir():
        ...
```

**After:**
```python
def get_train_test_splits(directory: Path = config.PATHS.TRAIN_TEST_SPLIT) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test data from split files.
    
    Args:
        directory: Path to directory containing split files
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as numpy arrays
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        NotADirectoryError: If path is not a directory
    """
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected directory, got file: {directory}")
    
    for file in directory.iterdir():
        ...
```

#### 3.1b: `drowsiness_detection/data.py` - `get_kss_labels_for_feature_file()`

**Before:**
```python
def get_kss_labels_for_feature_file(feature_file_path: Path) -> Union[np.ndarray, None]:
    """Load Karolinska Sleepiness Scale (KSS) labels for a feature file.
    
    Finds corresponding label file and extracts interpolated KSS values.
    
    Args:
        feature_file_path: Path to feature file
        
    Returns:
        Array of KSS labels or None if not found
    """
    interpolated_kss_index = 2
    identifier = str(feature_file_path.stem)[-11:]
    for label_file in config.PATHS.LABEL_DATA.iterdir():
        if identifier in str(label_file):
            return np.load(label_file)[:, interpolated_kss_index]
    else:
        return None
```

**After:**
```python
def get_kss_labels_for_feature_file(feature_file_path: Path) -> np.ndarray:
    """Load Karolinska Sleepiness Scale (KSS) labels for a feature file.
    
    Finds corresponding label file and extracts interpolated KSS values.
    
    Args:
        feature_file_path: Path to feature file
        
    Returns:
        Array of KSS labels
        
    Raises:
        FileNotFoundError: If corresponding label file is not found
    """
    if not feature_file_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file_path}")
    
    interpolated_kss_index = 2
    identifier = str(feature_file_path.stem)[-11:]
    
    for label_file in config.PATHS.LABEL_DATA.iterdir():
        if identifier in str(label_file):
            labels = np.load(label_file)
            return labels[:, interpolated_kss_index]
    
    raise FileNotFoundError(
        f"Could not find label file for feature file '{feature_file_path.name}'. "
        f"Searched for identifier '{identifier}' in {config.PATHS.LABEL_DATA}"
    )
```

#### 3.1c: `drowsiness_detection/models.py` - Add validation to model builders

Add parameter validation to `build_cnn_model()`:

**Before:**
```python
def build_cnn_model(input_shape: tuple, ...):
    """Build a 1D Convolutional Neural Network for time series classification."""
    input_layer = keras.layers.Input(input_shape[1:])
```

**After:**
```python
def build_cnn_model(input_shape: tuple, ...):
    """Build a 1D Convolutional Neural Network for time series classification.
    
    Args:
        input_shape: Shape tuple (batch_size, sequence_length, n_features)
        ...
        
    Raises:
        ValueError: If input_shape doesn't have 3 dimensions or if parameters are invalid
    """
    if len(input_shape) != 3:
        raise ValueError(
            f"input_shape must have 3 dimensions (batch, sequence_length, features), "
            f"got {len(input_shape)} dimensions: {input_shape}"
        )
    if num_filters <= 0:
        raise ValueError(f"num_filters must be positive, got {num_filters}")
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    if not 0 <= dropout_rate < 1:
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
    
    input_layer = keras.layers.Input(input_shape[1:])
```

Do the same for `build_lstm_model()` and `build_bi_lstm_model()`.

---

### Fix 3.2: Fix Return Type Annotations

**File:** `drowsiness_detection/data.py`  
**Function:** `get_train_test_splits()`

**Before:**
```python
def get_train_test_splits(directory: Path = config.PATHS.TRAIN_TEST_SPLIT) -> Tuple[np.ndarray, np.ndarray]:
    ...
    return X_train, y_train, X_test, y_test  # 4 returns!
```

**After:**
```python
def get_train_test_splits(
    directory: Path = config.PATHS.TRAIN_TEST_SPLIT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test data from split files.
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test) where:
        - X_train: Training features of shape (n_train_samples, n_features)
        - y_train: Training labels (binarized)
        - X_test: Test features of shape (n_test_samples, n_features)
        - y_test: Test labels (binarized)
    """
    ...
    return X_train, y_train, X_test, y_test
```

---

## Phase 4: Documentation and Type Hints (1-2 hours)

### Fix 4.1: Add Type Hints to Unannotated Parameters

**File:** `drowsiness_detection/helpers.py`

In `ArrayWrapper.__init__()`:

**Before:**
```python
def __init__(self, directory: Path, filename_generator, max_size_mb: int = 1, n_cols: int = 1):
```

**After:**
```python
from typing import Callable, Generator

def __init__(
    self,
    directory: Path,
    filename_generator: Callable[[], str],
    max_size_mb: int = 1,
    n_cols: int = 1
) -> None:
```

Update docstring:
```python
    """Initialize ArrayWrapper.
    
    Args:
        directory: Path to save chunk files
        filename_generator: Callable that yields unique filenames (e.g., generator function)
        max_size_mb: Maximum size per chunk in MB
        n_cols: Number of columns in array
    """
```

### Fix 4.2: Improve Empty Exception Messages

**File:** `drowsiness_detection/run_grid_search_experiment.py`

Find and replace empty `raise ValueError` statements:

**Before:**
```python
raise ValueError
```

**After (context-dependent):**
```python
raise ValueError(
    f"Configuration for model '{model_name}' not found in named configs. "
    f"Available models: {', '.join(available_models)}"
)
```

---

## Phase 5: Testing (3-4 hours)

### Fix 5.1: Create Tests for `data.py`

**New file:** `tests/test_data.py`

Create comprehensive tests for critical functions:

```python
"""Tests for drowsiness_detection.data module."""

import pytest
import numpy as np
from pathlib import Path
from drowsiness_detection import data, config


class TestGetKSSLabelsForFeatureFile:
    """Tests for get_kss_labels_for_feature_file function."""
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        fake_path = Path("/nonexistent/path/file.npy")
        with pytest.raises(FileNotFoundError):
            data.get_kss_labels_for_feature_file(fake_path)


class TestBinarizeTargets:
    """Tests for target binarization functions."""
    
    def test_discretize_binary(self):
        """Test binary discretization."""
        targets = np.array([4, 6, 8, 10])
        result = data.discretize_labels_by_threshold(targets, num_targets=2)
        # Values >= 7 become 1, < 7 become 0
        expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_discretize_three_class(self):
        """Test 3-class discretization."""
        targets = np.array([4, 6, 8, 10])
        result = data.discretize_labels_by_threshold(targets, num_targets=3)
        # Bins at [6, 8, 10], so: <6→0, 6-8→1, 8-10→2, ≥10→3
        assert result.shape == targets.shape
        assert len(np.unique(result)) <= 3


class TestTrainTestSplit:
    """Tests for train/test splitting functions."""
    
    def test_split_preserves_data_count(self):
        """Test that split preserves total sample count."""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)
        subject_data = np.column_stack([
            np.random.randint(1, 5, 100),
            np.random.randint(1, 20, 100)
        ])
        
        X_train, X_test, y_train, y_test, _, _ = data.train_test_split_by_subjects(
            X, y, num_targets=2, test_size=0.2, subject_data=subject_data
        )
        
        assert X_train.shape[0] + X_test.shape[0] == 100
        assert y_train.shape[0] + y_test.shape[0] == 100
```

### Fix 5.2: Fix Existing Test Isolation Issues

**File:** `tests/conftest.py` (create if doesn't exist)

```python
"""Pytest configuration and fixtures."""

import pytest
from drowsiness_detection import config


@pytest.fixture
def reset_paths():
    """Reset config paths after each test."""
    original_paths = config.PATHS
    yield
    config.PATHS = original_paths


@pytest.fixture
def setup_test_paths():
    """Set up standard test paths."""
    config.set_paths(1, 1)
    yield
    # Cleanup is handled by reset_paths fixture
```

Use in tests:
```python
def test_something(setup_test_paths):
    # Test code here uses configured paths
    pass
```

---

## Validation Checklist

After implementing each phase, verify:

### Phase 1 Checklist:
- [ ] Run `python -m pytest tests/ -v` - all tests pass
- [ ] No duplicate function definitions in models.py
- [ ] Import `get_train_test_splits` from data.py successfully
- [ ] No references to typo'd function name

### Phase 2 Checklist:
- [ ] `drowsiness_detection/logging_config.py` exists
- [ ] `drowsiness_detection/constants.py` exists and is imported
- [ ] All imports work: `python -c "from drowsiness_detection.logging_config import setup_logging; setup_logging()"`
- [ ] No `print()` statements in metrics.py (replaced with logging)

### Phase 3 Checklist:
- [ ] Run `python -c "from drowsiness_detection.data import get_train_test_splits; get_train_test_splits(Path('/nonexistent'))"`
- [ ] Should raise `FileNotFoundError` with descriptive message
- [ ] Type annotations all correct: `python -m mypy drowsiness_detection/ --ignore-missing-imports` passes

### Phase 4 Checklist:
- [ ] All public function parameters have type hints
- [ ] No empty exception messages
- [ ] Documentation is consistent and complete

### Phase 5 Checklist:
- [ ] `tests/test_data.py` exists with 10+ test cases
- [ ] All data.py functions have at least 1 test
- [ ] Run `python -m pytest tests/test_data.py -v` - all pass
- [ ] Test coverage improved: `python -m pytest --cov=drowsiness_detection tests/`

---

## Quick Implementation Summary

**Total changes by file:**

| File | Changes | Priority |
|------|---------|----------|
| `models.py` | Remove duplicates (lines 276-325) | High |
| `data.py` | Fix KSS_THRESHOLD, remove duplication, add validation | High |
| `helpers.py` | Fix typo in function name, add type hints | Medium |
| `visualize.py` | Remove unused import | Low |
| `constants.py` | Create new file with constants | High |
| `logging_config.py` | Create new file with logging setup | Medium |
| `metrics.py` | Replace print with logging | Medium |
| `conftest.py` | Create pytest fixtures | Medium |
| `test_data.py` | Create comprehensive tests | Medium |

---

## Commands to Verify All Fixes

After completing all phases:

```bash
# 1. Check imports work
python -c "from drowsiness_detection import config, data, models, helpers, constants; print('✓ All imports work')"

# 2. Run all tests
python -m pytest tests/ -v

# 3. Check for any remaining print statements
grep -r "print(" drowsiness_detection/ --include="*.py" | grep -v "test_" | grep -v ".pyc"

# 4. Type checking (if mypy installed)
python -m mypy drowsiness_detection/ --ignore-missing-imports

# 5. Code style (if flake8 installed)
python -m flake8 drowsiness_detection/ --max-line-length=100

# 6. Calculate test coverage
python -m pytest --cov=drowsiness_detection tests/ --cov-report=html
```

