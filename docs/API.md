# API Documentation

This document describes the main API of the drowsiness-detection package.

## Core Modules

### config.py

Configuration and path management.

#### `Paths` (class)
Data path management for all datasets.

**Attributes:**
- `DATA` - Base path for all data files
- `WINDOW_DATA` - Preprocessed window data
- `LABEL_DATA` - Label files
- `TRAIN_TEST_SPLIT` - Train/test split definitions
- `WINDOW_FEATURES` - Engineered features

**Example:**
```python
from drowsiness_detection.config import set_paths, PATHS

set_paths(frequency=1, seconds=1)
print(PATHS.WINDOW_DATA)
```

---

### models.py

Neural network architectures and model builders.

#### `ThreeDStandardScaler`
Custom scaler for 3D arrays.

**Methods:**
- `fit(X)` - Fit scaler on data
- `transform(X)` - Transform data using fitted scaler

**Example:**
```python
from drowsiness_detection.models import ThreeDStandardScaler
import numpy as np

scaler = ThreeDStandardScaler(feature_axis=2)
X = np.random.randn(100, 50, 32)  # (samples, time, features)
scaler.fit(X)
X_scaled = scaler.transform(X)
```

#### `build_cnn_model()`
Build a 1D Convolutional Neural Network.

**Parameters:**
- `input_shape` (tuple) - Input shape including batch dimension
- `kernel_size` (int) - Convolution kernel size
- `num_filters` (int) - Number of convolutional filters
- `num_conv_layers` (int) - Number of conv layers
- `use_batch_norm` (bool) - Whether to use batch normalization
- `dropout_rate` (float) - Dropout rate
- `learning_rate` (float) - Adam optimizer learning rate

**Returns:** Compiled Keras model

**Example:**
```python
from drowsiness_detection.models import build_cnn_model

model = build_cnn_model(
    input_shape=(None, 100, 64),
    kernel_size=5,
    num_filters=32,
    num_conv_layers=2
)
```

#### `build_lstm_model()`
Build an LSTM network.

**Parameters:**
- `input_shape` (tuple) - Input shape
- `lstm_units` (int) - Units per LSTM layer
- `num_lstm_layers` (int) - Number of LSTM layers
- `dropout_rate` (float) - Dropout rate
- `learning_rate` (float) - Learning rate

**Returns:** Compiled Keras model

#### `build_bi_lstm_model()`
Build a Bidirectional LSTM network.

**Parameters:** Same as `build_lstm_model()`

**Returns:** Compiled Keras model

---

### metrics.py

Classification evaluation metrics.

#### `calc_classification_metrics(y_trues, y_preds)`
Calculate metrics for binary classification.

**Parameters:**
- `y_trues` (list) - List of true labels or probabilities
- `y_preds` (list) - List of predicted labels or probabilities

**Returns:** Tuple of (accuracies, recalls, precisions, aucs)

**Example:**
```python
from drowsiness_detection.metrics import calc_classification_metrics
import numpy as np

y_true = [np.array([1, 1, 0, 0])]
y_pred = [np.array([0.9, 0.8, 0.1, 0.2])]

accs, recalls, precisions, aucs = calc_classification_metrics(y_true, y_pred)
print(f"Accuracy: {accs[0]:.3f}")
```

#### `calc_mean_and_std_of_classification_metrics(y_trues, y_preds)`
Calculate mean and standard deviation of metrics.

**Parameters:** Same as `calc_classification_metrics()`

**Returns:** Tuple of ((mean_acc, std_acc), (mean_recall, std_recall), ...)

#### `print_metric_results(results)`
Pretty-print metric results.

**Parameters:**
- `results` - Output from `calc_mean_and_std_of_classification_metrics()`

---

### data.py

Data loading and preprocessing.

#### `get_train_test_splits(directory)`
Load preprocessed train and test data.

**Parameters:**
- `directory` (Path) - Directory containing split data

**Returns:** Tuple of (test_data, train_data)

**Example:**
```python
from drowsiness_detection.data import get_train_test_splits
from drowsiness_detection.config import PATHS

test_data, train_data = get_train_test_splits(PATHS.TRAIN_TEST_SPLIT)
print(f"Test samples: {len(test_data)}")
print(f"Train samples: {len(train_data)}")
```

---

### helpers.py

Utility functions.

#### `binarize(arr, threshold)`
Convert continuous values to binary labels.

**Parameters:**
- `arr` (np.ndarray) - Input array
- `threshold` (float) - Threshold value

**Returns:** Binary array (1 if >= threshold, 0 otherwise)

#### `digitize(arr, shift, label_start_index)`
Digitize array values into discrete bins.

**Parameters:**
- `arr` (np.ndarray) - Input array
- `shift` (float) - Bin shift value
- `label_start_index` (int) - Starting index for labels

**Returns:** Digitized array

#### `label_to_one_hot_like(arr, k)`
Convert ordinal labels to one-hot-like encoding.

**Parameters:**
- `arr` (np.ndarray) - Ordinal labels
- `k` (int) - Maximum label value + 1

**Returns:** Binary matrix for ordinal regression

**Example:**
```python
from drowsiness_detection.helpers import label_to_one_hot_like
import numpy as np

labels = np.array([1, 2, 3, 4])
encoded = label_to_one_hot_like(labels, k=5)
print(encoded)
# [[0, 0, 0, 0],
#  [1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0]]
```

---

## Advanced Usage

### Custom Model Training

```python
from drowsiness_detection.models import build_lstm_model
from drowsiness_detection.data import get_train_test_splits
from drowsiness_detection.config import PATHS
import numpy as np

# Load data
test_data, train_data = get_train_test_splits(PATHS.TRAIN_TEST_SPLIT)
X_train = train_data[:, :-1]
y_train = train_data[:, -1]

# Build model
model = build_lstm_model(
    input_shape=X_train.shape,
    lstm_units=128,
    num_lstm_layers=2
)

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

### Hyperparameter Grid Search

See `drowsiness_detection/run_grid_search_experiment.py` for the complete grid search framework using Sacred.

---

## Further Reading

- [Setup Guide](SETUP.md) - Environment setup and configuration
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [README](../README.md) - Project overview
