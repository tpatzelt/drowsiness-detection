# Code Quality Analysis: Drowsiness Detection Repository

**Date:** February 3, 2026  
**Analysis Focus:** Code structure, patterns, organization, error handling, configuration, and testing

---

## Executive Summary

The drowsiness-detection repository contains 609+ lines of code across the main package with a publication-ready codebase. Analysis identified **26 specific issues** affecting code quality, maintainability, and robustness. Most issues are moderate-impact and can be resolved through targeted refactoring without major architectural changes.

---

## 1. CODE STRUCTURE & ORGANIZATION

### 1.1 DUPLICATE FUNCTION DEFINITIONS IN `models.py`

**Issue:** Functions `build_lstm_model` and `build_bi_lstm_model` are defined **twice** with identical or near-identical implementations.

**Location:** [drowsiness_detection/models.py](drowsiness_detection/models.py#L163-L200) (first) and [drowsiness_detection/models.py](drowsiness_detection/models.py#L276-L323) (duplicate)

**Impact:** High - Creates maintenance burden, confusion, and potential for divergent behavior if one is updated

**Recommendation:** 
- Remove lines 276-325 (the duplicate definitions)
- Keep the first, fully-documented versions (lines 163-245)
- Verify no code calls the duplicate definitions

**Exact Changes:**
```python
# DELETE lines 276-325 in models.py entirely
# This removes:
# - def build_lstm_model(input_shape, lstm_units=128, ...) [duplicate]
# - def build_bi_lstm_model(input_shape, lstm_units=128, ...) [duplicate]
# - def build_lstm_model(...) [another duplicate at end of file]
```

---

### 1.2 DUPLICATE LABEL DISCRETIZATION LOGIC

**Issue:** The logic for converting KSS labels to discrete targets exists in two separate functions with 90% code duplication.

**Locations:**
- [data.py#L227-L261](drowsiness_detection/data.py#L227-L261) - `preprocess_feature_data()` (inline logic)
- [data.py#L263-L282](drowsiness_detection/data.py#L263-L282) - `discretize_labels_by_threshold()` (extracted function)

**Impact:** Medium - Inconsistent application and maintenance burden

**Recommendation:** Replace inline logic in `preprocess_feature_data()` with call to `discretize_labels_by_threshold()`

**Exact Changes:**
```python
# In preprocess_feature_data() function, replace lines 234-252:
OLD:
    if num_targets == 2:
        KSS_THRESHOLD = 7
        targets = binarize(targets, threshold=KSS_THRESHOLD)
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

NEW:
    targets = discretize_labels_by_threshold(targets, num_targets=num_targets)
```

---

### 1.3 MONOLITHIC `data.py` FILE (609 LINES)

**Issue:** `data.py` contains data loading, preprocessing, train-test splitting, and experiment utilities in a single 609-line file. Functions lack clear separation of concerns.

**Impact:** Medium - Reduces readability and makes testing harder

**Recommendation:** Consider reorganizing into submodules:
```
drowsiness_detection/
├── data/
│   ├── __init__.py
│   ├── loaders.py          # get_train_test_splits(), load_nn_data(), get_feature_data()
│   ├── preprocessors.py    # preprocess_feature_data(), discretize_labels_by_threshold()
│   └── splitting.py        # train_test_split_by_subjects(), window_files_train_test_split()
```

**Alternative (minimal refactoring):** Add logical grouping comments and ensure related functions are adjacent.

---

### 1.4 LARGE FUNCTION WITH COMPLEX LOGIC

**Issue:** [`train_test_split_by_subjects()`](drowsiness_detection/data.py#L412-L466) is 54 lines with nested loops, histogram calculations, and complex conditional logic.

**Impact:** Medium - Difficult to test, maintain, and understand

**Recommendation:** Extract helper function for the stratification logic:
```python
def _compute_stratification_distances(train_histogram, test_histogram, new_labels, bins):
    """Compute which split (train/test) better maintains label distribution."""
    test_labels_for_dist = np.concatenate([test_histogram, new_labels])
    train_labels_for_dist = np.concatenate([train_histogram, new_labels])
    test_hist = np.histogram(test_labels_for_dist, bins=bins)[0]
    train_hist = np.histogram(train_labels_for_dist, bins=bins)[0]
    
    original_test_hist = np.histogram(test_histogram, bins=bins)[0]
    original_train_hist = np.histogram(train_histogram, bins=bins)[0]
    
    dist_if_train_added = np.linalg.norm(train_hist - original_test_hist)
    dist_if_test_added = np.linalg.norm(test_hist - original_train_hist)
    
    return dist_if_train_added, dist_if_test_added
```

Then simplify the main loop to call this function.

---

## 2. CODE PATTERNS & ANTI-PATTERNS

### 2.1 MAGIC NUMBERS AND HARDCODED THRESHOLDS (HIGH PRIORITY)

**Issue:** Multiple hardcoded numeric values scattered throughout without clear documentation or configuration.

**Specific Examples:**

1. **KSS_THRESHOLD = 7** appears 3 times:
   - [data.py#L151](drowsiness_detection/data.py#L151) - undefined variable, should fail
   - [data.py#L235](drowsiness_detection/data.py#L235)
   - [data.py#L265](drowsiness_detection/data.py#L265)

2. **Feature indices magic numbers**: [data.py#L288](drowsiness_detection/data.py#L288) and [run_grid_search_experiment.py#L75](drowsiness_detection/run_grid_search_experiment.py#L75)
   - `(5, 8, 9, 14, 15, 16, 19)` appears in multiple places without explanation

3. **Threshold bins** hardcoded in multiple functions:
   - ALERT = 6, NEUTRAL = 8, SLEEPY = 10 [data.py#L241-L243](drowsiness_detection/data.py#L241-L243)
   - `[3, 5, 7, 9]` for 5-target classification [data.py#L246-L250](drowsiness_detection/data.py#L246-L250)

4. **Neural network hyperparameters** hardcoded:
   - Input shapes: `(20, 300, 7)` [run_grid_search_experiment.py#L173](drowsiness_detection/run_grid_search_experiment.py#L173)
   - Dense layer units: `32` [models.py#L158](drowsiness_detection/models.py#L158)
   - Learning rates: `0.002` [models.py#L133](drowsiness_detection/models.py#L133), `0.001` [models.py#L182](drowsiness_detection/models.py#L182)

**Impact:** High - Makes experimentation difficult, inconsistency risks, harder to reproduce results

**Recommendation:** Create a configuration constants module:

```python
# drowsiness_detection/constants.py
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

# Neural Network defaults
NN_DEFAULT_INPUT_SHAPE = (20, 300, 7)
NN_DEFAULT_DENSE_UNITS = 32
NN_DEFAULT_LEARNING_RATE = 0.001

# File size limits
MAX_CHUNK_SIZE_MB = 100
```

Then update code to use:
```python
# Before
y_train = binarize(y_train, KSS_THRESHOLD)

# After
from drowsiness_detection.constants import KSS_THRESHOLD_BINARY
y_train = binarize(y_train, KSS_THRESHOLD_BINARY)
```

---

### 2.2 UNDEFINED VARIABLE BUG

**Issue:** [data.py#L151](drowsiness_detection/data.py#L151) uses undefined variable `KSS_THRESHOLD`

**Location:** [data.py](drowsiness_detection/data.py#L130-L152)

**Current Code:**
```python
def get_train_test_splits(directory: Path = config.PATHS.TRAIN_TEST_SPLIT) -> Tuple[np.ndarray, np.ndarray]:
    """Load train and test data from split files.
    
    Binarizes targets using KSS_THRESHOLD of 7.
    ...
    """
    # ... code ...
    y_train, y_test = binarize(y_train, KSS_THRESHOLD), binarize(y_test, KSS_THRESHOLD)
    # NameError: name 'KSS_THRESHOLD' is not defined
```

**Fix:**
```python
from drowsiness_detection.constants import KSS_THRESHOLD_BINARY

def get_train_test_splits(directory: Path = config.PATHS.TRAIN_TEST_SPLIT) -> Tuple[np.ndarray, np.ndarray]:
    """Load train and test data from split files.
    
    Binarizes targets using KSS_THRESHOLD of 7.
    """
    # ... code ...
    y_train, y_test = binarize(y_train, KSS_THRESHOLD_BINARY), binarize(y_test, KSS_THRESHOLD_BINARY)
```

---

### 2.3 PRINT STATEMENTS INSTEAD OF LOGGING (MEDIUM PRIORITY)

**Issue:** Print statements used throughout codebase instead of Python logging module, making it impossible to:
- Control verbosity at runtime
- Redirect to files
- Set different log levels for different modules
- Integrate with monitoring systems

**Locations (20+ occurrences):**
- [metrics.py#L65-68](drowsiness_detection/metrics.py#L65-68) - `print_metric_results()` function
- [models.py#L75, 99, 159, 200, 245, 272, 297, 323](drowsiness_detection/models.py) - Model builder functions
- [helpers.py#L28](drowsiness_detection/helpers.py#L28) - `print_nan_intersections()`
- [visualize.py#L357, 361, 371, etc.](drowsiness_detection/visualize.py) - Multiple visualization functions
- [run_grid_search_experiment.py#L423](drowsiness_detection/run_grid_search_experiment.py#L423)

**Recommendation:** Add logging configuration:

```python
# drowsiness_detection/logging_config.py
"""Logging configuration for drowsiness detection."""
import logging

def setup_logging(level=logging.INFO, log_file=None):
    """Configure logging for the package."""
    logger = logging.getLogger('drowsiness_detection')
    logger.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

# In each module:
import logging
logger = logging.getLogger(__name__)
```

Replace specific examples:
```python
# metrics.py - Before
def print_metric_results(results: tuple) -> None:
    results = [(round(x, 2), round(y, 3)) for x, y in results]
    (mean_acc, std_acc), ... = results
    print(rf"Mean Accuracy = {mean_acc} ± {std_acc}")
    print(rf"Mean Precision = {mean_precision} ± {std_precision}")

# After
import logging
logger = logging.getLogger(__name__)

def print_metric_results(results: tuple) -> None:
    results = [(round(x, 2), round(y, 3)) for x, y in results]
    (mean_acc, std_acc), ... = results
    logger.info(f"Mean Accuracy = {mean_acc} ± {std_acc}")
    logger.info(f"Mean Precision = {mean_precision} ± {std_precision}")
```

---

### 2.4 INCONSISTENT PARAMETER ANNOTATIONS

**Issue:** Functions use inconsistent type hints - some have full annotations, some partial, some none.

**Examples:**
- [helpers.py#L100](drowsiness_detection/helpers.py#L100) - Missing type hints:
  ```python
  def __init__(self, directory: Path, filename_generator, max_size_mb: int = 1, n_cols: int = 1):
  # filename_generator is not annotated
  ```

- [models.py#L276, 301](drowsiness_detection/models.py#L276-L325) - Duplicate functions missing all type hints:
  ```python
  def build_lstm_model(input_shape, lstm_units=128, learning_rate=.001, ...):
      # Should be: def build_lstm_model(input_shape: tuple, lstm_units: int = 128, ...)
  ```

**Impact:** Low - Reduces code clarity for type-aware IDEs

**Recommendation:** Add consistent type hints using imports from `typing`:

```python
from typing import Callable, Generator
from pathlib import Path
import numpy as np

def __init__(
    self, 
    directory: Path, 
    filename_generator: Callable[[], str],  # Generator function
    max_size_mb: int = 1, 
    n_cols: int = 1
) -> None:
```

---

### 2.5 WEAK EXCEPTION HANDLING

**Issue:** Several functions raise exceptions without descriptive error messages or insufficient validation.

**Examples:**

1. [data.py#L96](drowsiness_detection/data.py#L96):
   ```python
   if not target_dir.exists():
       target_dir.mkdir()
   else:
       raise RuntimeError("directory exists.")  # Generic, unclear error
   ```
   Should be:
   ```python
   if target_dir.exists():
       raise FileExistsError(
           f"Target directory already exists: {target_dir}. "
           f"Please remove it or choose a different path."
       )
   ```

2. [run_grid_search_experiment.py#L318, 328, 340](drowsiness_detection/run_grid_search_experiment.py#L318):
   ```python
   raise ValueError  # No message!
   ```
   Should include context.

3. [visualize.py#L316](drowsiness_detection/visualize.py#L316):
   ```python
   raise AttributeError("unknown grid")  # Should be ValueError
   ```

---

## 3. PACKAGE ORGANIZATION

### 3.1 GLOBAL STATE IN `config.py`

**Issue:** Module-level global variable `PATHS` modified by function `set_paths()`, making it non-reentrant and causing issues with parallel/concurrent code.

**Locations:** [config.py#L10, 64](drowsiness_detection/config.py#L10-L64)

```python
# Global variable
PATHS = None

def set_paths(frequency: int, seconds: int):
    """Switch function to change data sources."""
    global PATHS  # Using global, which is problematic
    PATHS = Paths(frequency=frequency, seconds=seconds)

set_paths(1, 1)  # Set at module import time
```

**Problems:**
- Not thread-safe
- Hard to test (state persists across tests)
- Implicit dependency on module initialization order

**Recommendation:** Use dependency injection pattern:

```python
# Option 1: Create path manager class
class PathManager:
    """Manages data paths for different configurations."""
    
    def __init__(self, frequency: int = 1, seconds: int = 1):
        self.paths = Paths(frequency=frequency, seconds=seconds)
    
    def set_paths(self, frequency: int, seconds: int) -> None:
        """Update paths."""
        self.paths = Paths(frequency=frequency, seconds=seconds)
    
    def get_paths(self) -> Paths:
        return self.paths

# Usage
path_manager = PathManager()
data_loader = DataLoader(path_manager)

# Option 2: Use context manager
@contextmanager
def paths_context(frequency: int, seconds: int):
    """Context manager for path configuration."""
    global PATHS
    old_paths = PATHS
    try:
        PATHS = Paths(frequency=frequency, seconds=seconds)
        yield PATHS
    finally:
        PATHS = old_paths

# Usage
with paths_context(30, 60):
    # Use PATHS within this context
    data = load_data()
```

**For now (minimal change):** At least document the issue:

```python
# Temporary global state - not thread-safe
# TODO: Refactor to use dependency injection
PATHS = None
```

---

### 3.2 UNUSED IMPORT

**Issue:** `deepcopy` imported but never used in [visualize.py#L1](drowsiness_detection/visualize.py#L1)

**Recommendation:** Remove line 1: `from copy import deepcopy`

---

### 3.3 INCOMPLETE TEST FILES

**Issue:** Tests exist for `metrics.py` and `models.py`, but:
- No tests for `data.py` (609 lines, most complex module)
- No tests for `helpers.py` (205 lines)
- No tests for `visualize.py` (741 lines)
- No tests for `config.py`

**Impact:** High - Most critical data loading/processing code untested

**Current test coverage:**
- `test_helpers.py`: 76 lines, covers ~50% of helpers.py (only binarize, digitize, label_to_one_hot_like)
- `test_metrics.py`: 79 lines, covers calc_classification_metrics functions
- `test_models.py`: 86 lines, covers model building functions

**Missing test coverage:**
- All data loading/preprocessing in `data.py` (functions like `get_train_test_splits()`, `preprocess_feature_data()`, etc.)
- Configuration system in `config.py`
- `ArrayWrapper` class in `helpers.py`
- All visualization functions in `visualize.py`

---

## 4. ERROR HANDLING & VALIDATION

### 4.1 MISSING INPUT VALIDATION

**Issue:** Functions don't validate inputs, leading to cryptic errors downstream.

**Examples:**

1. [data.py#L156](drowsiness_detection/data.py) - `get_train_test_splits()`:
   ```python
   def get_train_test_splits(directory: Path = config.PATHS.TRAIN_TEST_SPLIT):
       for file in directory.iterdir():  # Will crash if directory doesn't exist
   ```
   Should validate:
   ```python
   def get_train_test_splits(directory: Path = config.PATHS.TRAIN_TEST_SPLIT):
       if not directory.exists():
           raise FileNotFoundError(f"Data directory not found: {directory}")
       if not directory.is_dir():
           raise NotADirectoryError(f"Expected directory, got file: {directory}")
   ```

2. [models.py#L103](drowsiness_detection/models.py) - `build_cnn_model()`:
   ```python
   def build_cnn_model(input_shape: tuple, ...):
       # Doesn't validate input_shape has 3 dimensions
   ```
   Should add:
   ```python
   if len(input_shape) != 3:
       raise ValueError(f"input_shape must have 3 dimensions (batch, time, features), got {len(input_shape)}")
   ```

3. [data.py#L207](drowsiness_detection/data.py) - `drop_by_identifier()`:
   ```python
   def drop_by_identifier(X, y, identifiers: np.ndarray, exclude_by: int):
       if X.shape[0] != identifiers.shape[0] or y.shape[0] != identifiers.shape[0]:
           raise ValueError("Shapes do not match.")  # Only checks after calling function
   ```

---

### 4.2 SILENT FAILURES

**Issue:** Functions can fail silently without clear error messages.

**Example:** [data.py#L63-68](drowsiness_detection/data.py#L63-68):
```python
def get_kss_labels_for_feature_file(feature_file_path: Path):
    """Load Karolinska Sleepiness Scale (KSS) labels for a feature file."""
    identifier = str(feature_file_path.stem)[-11:]
    for label_file in config.PATHS.LABEL_DATA.iterdir():
        if identifier in str(label_file):
            return np.load(label_file)[:, interpolated_kss_index]
    else:
        return None  # Silently returns None if label not found
```

**Better:**
```python
def get_kss_labels_for_feature_file(feature_file_path: Path):
    """Load Karolinska Sleepiness Scale (KSS) labels for a feature file.
    
    Raises:
        FileNotFoundError: If corresponding label file is not found
    """
    identifier = str(feature_file_path.stem)[-11:]
    for label_file in config.PATHS.LABEL_DATA.iterdir():
        if identifier in str(label_file):
            return np.load(label_file)[:, interpolated_kss_index]
    
    raise FileNotFoundError(
        f"Could not find label file for feature file {feature_file_path.name}. "
        f"Searched for identifier '{identifier}' in {config.PATHS.LABEL_DATA}"
    )
```

---

## 5. DEPENDENCIES & IMPORTS

### 5.1 IMPORT ORGANIZATION

**Issue:** Imports not organized according to PEP 8 standard (standard library, third-party, local).

**Example - [visualize.py](drowsiness_detection/visualize.py#L1-L15):**
```python
from copy import deepcopy  # Standard lib
from pathlib import Path   # Standard lib
import matplotlib.patches as mpatches  # Third-party
import matplotlib.pyplot as plt        # Third-party
import numpy as np                     # Third-party
import pandas as pd                    # Third-party
from matplotlib import animation       # Third-party
# ... more third-party imports ...
from drowsiness_detection import config  # Local
from drowsiness_detection.data import ...  # Local
```

**Recommended:**
```python
# Standard library
from copy import deepcopy
from pathlib import Path

# Third-party
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
# ... rest of third-party ...
from sklearn.metrics import ...

# Local
from drowsiness_detection import config
from drowsiness_detection.data import ...
```

---

### 5.2 UNUSED IMPORT

**Location:** [visualize.py#L1](drowsiness_detection/visualize.py#L1)
```python
from copy import deepcopy  # Never used
```

**Action:** Remove this line.

---

### 5.3 VERSION SPECIFICATION ISSUES

**Issue:** `requirements.txt` uses approximate version pins (`~=`) which is not strict enough for production reproducibility.

**Current:** [requirements.txt](requirements.txt)
```
tensorflow~=2.8.0     # Allows 2.8.x but not 2.9+
scikit-learn~=1.0.2   # Allows 1.0.x but not 1.1+
```

**Concerns:**
- `tensorflow~=2.8.0` could update to 2.8.5, 2.8.10, etc., potentially with breaking changes
- Doesn't guarantee reproducible builds across different times/environments

**Recommendation:** Use exact versions or at least major.minor pinning:
```
tensorflow==2.8.0
scikit-learn==1.0.2
```

Or if flexibility is needed:
```
tensorflow>=2.8.0,<2.9.0
scikit-learn>=1.0.2,<1.1.0
```

---

## 6. CONFIGURATION MANAGEMENT

### 6.1 HARDCODED CONFIGURATION IN RUN SCRIPTS

**Issue:** [run.sh](run.sh) contains commented-out hardcoded experiment commands with magic values.

**Current:**
```bash
#python drowsiness_detection/run_grid_search_experiment.py with random_forest recording_frequency=30 window_in_sec=60 grid_search_params.n_jobs=-1 num_targets=9;
#python drowsiness_detection/run_grid_search_experiment.py with random_forest recording_frequency=30 window_in_sec=10 grid_search_params.n_jobs=-1 num_targets=2 seed=42;
```

**Issues:**
- Magic numbers scattered (30, 60, 10, 42, 9, 2)
- No explanation of what these parameters mean
- Hard to discover valid parameter combinations

**Recommendation:** Create a configuration file:
```yaml
# config/experiments.yaml
experiments:
  rf_60sec_9targets:
    model: random_forest
    recording_frequency: 30
    window_in_sec: 60
    num_targets: 9
    grid_search_params:
      n_jobs: -1
      
  rf_10sec_binary:
    model: random_forest
    recording_frequency: 30
    window_in_sec: 10
    num_targets: 2
    seed: 42
    grid_search_params:
      n_jobs: -1
```

---

### 6.2 MIXED CONFIGURATION STYLES

**Issue:** Configuration specified in three different ways:
1. Sacred config decorators [run_grid_search_experiment.py#L32-50](drowsiness_detection/run_grid_search_experiment.py#L32-50)
2. Named configs [run_grid_search_experiment.py#L53+](drowsiness_detection/run_grid_search_experiment.py#L53+)
3. Command-line overrides in [run.sh](run.sh)

**Recommendation:** Document configuration precedence clearly in README.

---

## 7. TESTING & QUALITY

### 7.1 INCOMPLETE TEST COVERAGE

**Summary Table:**

| Module | Lines | Test Lines | Tested Functions | Missing Coverage |
|--------|-------|-----------|------------------|-----------------|
| helpers.py | 205 | 76 | 3 | ArrayWrapper, name_generator, spec_to_config_space |
| metrics.py | ~70 | 79 | 2 | print_metric_results() |
| models.py | 325 | 86 | 4 | ThreeDStandardScaler, build_* functions |
| data.py | 609 | 0 | 0 | **ALL UNTESTED** - Most critical module |
| config.py | 64 | 0 | 0 | **UNTESTED** |
| visualize.py | 741 | 0 | 0 | **UNTESTED** |
| **TOTAL** | **2014** | **241** | **~9** | **~80% coverage missing** |

---

### 7.2 TEST ISOLATION ISSUES

**Issue:** Tests may not be isolated due to global state in `config.PATHS`.

**Example - [tests/test_models.py](tests/test_models.py):**
```python
def test_cnn_model_creation(self):
    input_shape = (None, 100, 64)
    model = build_cnn_model(input_shape)  # Works fine
```

But if `config.PATHS` isn't properly initialized, some functions could fail.

**Recommendation:** Use fixtures:
```python
import pytest
from drowsiness_detection import config

@pytest.fixture
def setup_paths():
    """Set up test configuration."""
    original_paths = config.PATHS
    config.set_paths(1, 1)
    yield
    config.PATHS = original_paths

def test_something(setup_paths):
    # Test code here
```

---

### 7.3 MISSING DOCSTRING EXAMPLES

**Issue:** Complex functions lack usage examples in docstrings.

**Example - [data.py#L412-L466](drowsiness_detection/data.py#L412-L466) - `train_test_split_by_subjects()`:
```python
def train_test_split_by_subjects(X, y, num_targets, test_size, subject_data):
    """Split data by subjects ensuring balanced class distribution.
    
    Args:
        X: Feature matrix
        y: Labels
        num_targets: Number of target classes
        test_size: Proportion for test set
        subject_data: Subject identifiers
        
    Returns:
        Tuple of (train, test, train_labels, test_labels, (train_ids, test_ids), ...)
    """
```

**Better:**
```python
def train_test_split_by_subjects(X, y, num_targets, test_size, subject_data):
    """Split data by subjects ensuring balanced class distribution.
    
    Uses stratification based on class distribution to ensure train/test
    sets have similar label distributions, and stratifies by subject
    to avoid subject leakage.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
        num_targets: Number of target classes (2, 3, 5, or 9)
        test_size: Proportion of data for test set
        subject_data: Subject identifiers of shape (n_samples, 2) with
                      [session_type, subject_id]
    
    Returns:
        Tuple of:
        - train: Training features
        - test: Test features
        - train_labels: Training labels
        - test_labels: Test labels
        - (train_ids, test_ids): Subject IDs for each set
        - (train_subject_info, test_subject_info): Full subject information
    
    Example:
        >>> X = np.random.randn(100, 50)
        >>> y = np.random.randint(0, 2, 100)
        >>> subject_data = np.column_stack([np.random.randint(1, 5, 100),
        ...                                  np.random.randint(1, 20, 100)])
        >>> train, test, y_train, y_test, ids, info = train_test_split_by_subjects(
        ...     X, y, num_targets=2, test_size=0.2, subject_data=subject_data
        ... )
        >>> train.shape[0] + test.shape[0] == 100
        True
    """
```

---

## 8. STYLE & NAMING CONVENTIONS

### 8.1 INCONSISTENT NAMING IN PARAMETER DEFAULTS

**Issue:** Parameter default values inconsistent between similar functions.

**Example - Model builders:**
```python
# LSTM defaults
def build_lstm_model(
    input_shape: tuple,
    lstm_units: int = 128,
    learning_rate: float = 0.001,      # float
    dropout_rate: float = 0.2,
    num_lstm_layers: int = 2):

# CNN has same pattern
def build_cnn_model(
    input_shape: tuple,
    kernel_size: int = 5,
    stride: int = 1,
    num_filters: int = 32,
    num_conv_layers: int = 2,
    padding: str = "same",
    use_batch_norm: bool = True,
    pooling: str = "average",
    dropout_rate: float = 0.2,
    learning_rate: float = 0.002):     # float but different value
```

**Inconsistency:** Different default learning rates (0.001 vs 0.002) without explanation.

**Recommendation:** Use constants:
```python
from drowsiness_detection.constants import (
    NN_DEFAULT_LEARNING_RATE,
    NN_DEFAULT_LSTM_UNITS,
    NN_DEFAULT_DROPOUT,
)

def build_lstm_model(
    input_shape: tuple,
    lstm_units: int = NN_DEFAULT_LSTM_UNITS,
    learning_rate: float = NN_DEFAULT_LEARNING_RATE,
    dropout_rate: float = NN_DEFAULT_DROPOUT,
    num_lstm_layers: int = 2):
```

---

### 8.2 TYPO IN FUNCTION NAME

**Issue:** Function named `create_emtpy_array_of_max_size()` with typo "emtpy" instead of "empty".

**Location:** [helpers.py#L128](drowsiness_detection/helpers.py#L128)

**Impact:** Low but unprofessional

**Fix:** Rename to `create_empty_array_of_max_size()` and update all references.

---

## 9. SPECIFIC CODE ISSUES REQUIRING FIXES

### Issue #1: Empty Exception Messages
**File:** [run_grid_search_experiment.py#L318, 328, 340](drowsiness_detection/run_grid_search_experiment.py)
```python
raise ValueError  # No message
```
**Fix:** Add descriptive messages
```python
raise ValueError(f"Model '{model_name}' is not supported")
```

---

### Issue #2: Return Type Inconsistency
**File:** [data.py#L130-L152](drowsiness_detection/data.py#L130-L152)
```python
def get_train_test_splits(...) -> Tuple[np.ndarray, np.ndarray]:
    """..."""
    # Returns 4 arrays, not 2!
    return X_train, y_train, X_test, y_test  # Mismatch with type hint
```
**Fix:** Correct type hint:
```python
def get_train_test_splits(...) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

---

### Issue #3: Incomplete Comment
**File:** [run_grid_search_experiment.py#L180](drowsiness_detection/run_grid_search_experiment.py)
```python
# "return_train_score": True
```
Should explain why it's commented out.

---

### Issue #4: Typo in Variable Name
**File:** [run_grid_search_experiment.py#L406](drowsiness_detection/run_grid_search_experiment.py)
```python
class_weights = fit_params.get("classifier__class_weight", None)
```
Then never used. Likely incomplete feature.

---

## PRIORITY MATRIX

### High Priority (Should Fix)
1. **Duplicate function definitions** in models.py (lines 276-325)
2. **Undefined variable KSS_THRESHOLD** in data.py#L151
3. **Missing test coverage** for data.py, config.py, visualize.py
4. **Magic numbers** scattered throughout (KSS thresholds, feature indices, etc.)
5. **Global state** in config.py (thread-safety concerns)

### Medium Priority (Should Consider)
1. Duplicate discretization logic in data.py
2. Print statements should use logging
3. Weak exception handling and missing validation
4. Monolithic data.py file (refactor into submodules)
5. Typo: `create_emtpy_array_of_max_size`
6. Large, complex functions (train_test_split_by_subjects)

### Low Priority (Nice to Have)
1. Import organization according to PEP 8
2. Unused imports (deepcopy in visualize.py)
3. Inconsistent type annotations
4. Version pinning in requirements.txt
5. Documentation improvements (examples in docstrings)

---

## SUMMARY OF RECOMMENDED CHANGES

**Count by Category:**
- Code duplication: 3 instances
- Magic numbers/hardcoded values: 8+ instances
- Print statements to replace: 20+ instances
- Missing tests: 3 major modules
- Missing input validation: 10+ functions
- Naming/typo issues: 2 instances
- Configuration issues: 3 instances
- Error handling improvements: 5+ locations

**Estimated effort to implement all recommendations:**
- High priority: 4-6 hours
- Medium priority: 8-12 hours
- Low priority: 3-5 hours
- **Total: 15-23 hours** (manageable, incremental improvement)

---

## NEXT STEPS

1. **Immediate:** Fix critical bugs (KSS_THRESHOLD, duplicate functions)
2. **Short-term:** Implement constants module and logging
3. **Medium-term:** Improve test coverage, add input validation
4. **Long-term:** Consider package restructuring (submodules for data/)

