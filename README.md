# Drowsiness Detection: Neural Networks vs. Engineered Features

> **Comparing the effectiveness of machine-learned features with hand-crafted features for detecting student drowsiness using eye-tracking data.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

## Overview

This repository contains the code and experimental framework for research comparing neural networks (CNNs, LSTMs) with traditional machine learning models (Random Forest) and engineered features for drowsiness detection. The project uses eye-tracking and eye-closure signals from students in baseline and sleep-deprived conditions.

**Research Question:** Can neural networks learn better features than carefully engineered hand-crafted features for detecting drowsiness?

### Key Features

- 📊 Multiple model architectures: Random Forest, CNN, LSTM, Bidirectional LSTM
- 🔬 Comprehensive hyperparameter grid search with Sacred experiment tracking
- 📈 Evaluation metrics including ROC curves, learning curves, and classification reports
- 🎯 Support for both engineered and learned features
- ⚙️ Configurable data preprocessing pipeline
- 🧪 Reproducible experimental framework with detailed logging

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Setup](#data-setup)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda for package management

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tpatzelt/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure data path:**
   Set the path to your drowsiness dataset:
   ```bash
   export DROWSINESS_DATA_PATH="/path/to/your/drowsiness_data"
   ```
   Or edit `drowsiness_detection/config.py` and set `BASE_PATH` directly.

### Running a Quick Experiment

```bash
# Run grid search experiments (see run.sh for examples)
python drowsiness_detection/run_grid_search_experiment.py \
    --config config_example.json \
    --seed 42

# View results
python notebooks/analysis/create_roc_curve_from_predictions.ipynb
```

## Installation

### Requirements

- **Python:** 3.8 or higher
- **Dependencies:** See [requirements.txt](requirements.txt)

### Development Setup

For development and running notebooks:

```bash
pip install -r requirements.txt
pip install jupyter jupyterlab pytest black flake8
```

### Configuration

The project requires access to preprocessed drowsiness data. Before running experiments:

1. **Prepare your data:**
   - See [Data Setup](#data-setup) section below
   - Run the preprocessing notebook: `notebooks/create_preprocessed_data.ipynb`

2. **Set data path (one of two ways):**
   
   **Option A: Environment variable (recommended)**
   ```bash
   export DROWSINESS_DATA_PATH="/path/to/drowsiness_data"
   ```
   
   **Option B: Direct configuration**
   ```python
   # In your script, before importing drowsiness_detection modules:
   from drowsiness_detection.config import BASE_PATH
   BASE_PATH = Path("/your/data/path")
   ```

## Project Structure

```
drowsiness-detection/
├── drowsiness_detection/          # Main package
│   ├── __init__.py
│   ├── config.py                  # Configuration and path management
│   ├── data.py                    # Data loading and preprocessing
│   ├── helpers.py                 # Utility functions
│   ├── metrics.py                 # Evaluation metrics
│   ├── models.py                  # Model architectures
│   ├── visualize.py               # Visualization functions
│   └── run_grid_search_experiment.py  # Experiment runner
├── notebooks/
│   ├── 1_data_exploration/        # Initial data exploration
│   ├── 2_preprocessing/           # Data preprocessing
│   │   └── create_preprocessed_data.ipynb
│   ├── 3_feature_engineering/     # Feature extraction
│   └── 4_results_analysis/        # Result analysis and visualization
├── logs_to_keep/                  # Archived experiment results
│   ├── log_summary.csv           # Summary of key experiments
│   └── log_notes.txt             # Notes on experiments
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
├── README.md                      # This file
└── run.sh                         # Example experiment commands
```

### File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration for paths, hyperparameters, and constants |
| `data.py` | Data loading, splitting, and preprocessing functions |
| `models.py` | Neural network architecture builders (CNN, LSTM, BiLSTM) and custom scalers |
| `metrics.py` | Classification metrics (accuracy, precision, recall, ROC curves) |
| `helpers.py` | General utility functions |
| `visualize.py` | Plotting functions for learning curves, ROC curves, and results |
| `run_grid_search_experiment.py` | Sacred-based experiment runner for hyperparameter search |

## Data Setup

### Dataset Requirements

This project uses eye-tracking and eye-closure signals from students. The expected data structure is:

```
drowsiness_data/
└── Windows_1_Hz/
    ├── WindowData/
    │   ├── 1_sec/
    │   │   ├── subject_001_baseline.npy
    │   │   └── ...
    │   └── Format/
    │       └── data_format.json
    ├── WindowLabels/
    │   └── KSS/
    │       └── 1_sec/
    │           └── subject_001_baseline_labels.npy
    ├── WindowFeatures/
    │   ├── 1_sec/
    │   │   ├── subject_001_baseline_features.npy
    │   │   └── ...
    │   └── feature_names.txt
    └── TrainTestSplits/
        └── 1_sec/
            └── [train/test split definitions]
```

### Preprocessing

To prepare raw data for experiments:

1. **Open the preprocessing notebook:**
   ```bash
   jupyter notebook notebooks/create_preprocessed_data.ipynb
   ```

2. **Follow the notebook:**
   - Set the input raw data path in the first cell
   - Run all cells to generate preprocessed data
   - Output will be saved to `data/preprocessed/`

### Data Classes

See `drowsiness_detection/data.py` for:
- `RowData` - Represents a single row of multimodal eye-tracking data
- `WindowDataset` - Manages train/test splits and feature preparation

## Usage

### Running Experiments

#### Single Experiment

```bash
python drowsiness_detection/run_grid_search_experiment.py \
    --config_file experiment_config.json \
    --seed 42 \
    --experiment_name my_experiment
```

#### Grid Search

See [run.sh](run.sh) for example grid search commands:

```bash
# Example: Search over multiple random forest hyperparameters
python drowsiness_detection/run_grid_search_experiment.py \
    --config config_rf_depth_samples.json
```

### Configuration Format

Experiments are configured via JSON files. Example:

```json
{
  "model": "random_forest",
  "model_params": {
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "random_state": 42
  },
  "data_params": {
    "frequency": 1,
    "seconds": 1,
    "train_test_split": 0.2,
    "use_engineered_features": true
  },
  "training_params": {
    "random_seed": 42
  }
}
```

### Using Sacred for Experiment Tracking

The project uses [Sacred](https://github.com/IDSIA/sacred) for experiment management:

```python
from sacred import Experiment

ex = Experiment('drowsiness_detection')

@ex.config
def my_config():
    model = 'random_forest'
    max_depth = 10

@ex.automain
def run(model, max_depth):
    # Your code here
    pass
```

Results are automatically logged to the database with configuration, metrics, and timing information.

## Results

### Experiment Structure

Results from experiments are stored in `logs_to_keep/` with the following structure:

```
logs_to_keep/
├── 102/
│   ├── config.json           # Experiment configuration
│   ├── info.json            # Metadata (timestamp, status)
│   ├── metrics.json         # Evaluation metrics
│   ├── train_history.csv    # Training history (if neural network)
│   └── best_model/          # Saved model weights (if applicable)
└── ...
```

### Key Results

See [logs_to_keep/log_summary.csv](logs_to_keep/log_summary.csv) for a summary of key experiment results.

Findings are documented in [logs_to_keep/log_notes.txt](logs_to_keep/log_notes.txt).

### Analyzing Results

Open analysis notebooks:

```bash
jupyter notebook notebooks/results_analysis/
```

Example notebooks:
- `create_roc_curve_from_predictions.ipynb` - ROC curve analysis
- `learning_curve_analysis.ipynb` - Model learning curves
- `feature_importance_analysis.ipynb` - Random Forest feature importance

## Model Architectures

### Random Forest
- Baseline model using scikit-learn
- Works with engineered features
- Fast training, interpretable

### CNN (1D Convolutional Neural Network)
```
Input → Conv1D → BatchNorm → ReLU → GlobalAvgPooling → Dense → Output
```
- Learns spatial patterns in time series
- Configurable kernel size, filters, and conv layers
- Uses sigmoid activation for binary classification

### LSTM (Long Short-Term Memory)
```
Input → LSTM(s) → Flatten → Dropout → Dense → Output
```
- Captures temporal dependencies
- Single or multi-layer configuration
- Bidirectional variant (BiLSTM) processes sequences in both directions

### BiLSTM (Bidirectional LSTM)
- Combines forward and backward LSTM layers
- Better context understanding for sequence modeling
- Recommended for eye-tracking time series

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Code Style:** Follow PEP 8; use Black for formatting
3. **Tests:** Add tests for new functionality
4. **Documentation:** Update docstrings and README as needed
5. **Commit Messages:** Use clear, descriptive messages
6. **Pull Request:** Describe changes and reference related issues

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Format code
black drowsiness_detection/

# Check for style issues
flake8 drowsiness_detection/

# Run tests
pytest tests/
```

## Citing This Work

If you use this code in your research, please cite:

```bibtex
@mastersthesis{yourname2024drowsiness,
  title={Can Neural Networks Beat Engineered Features at Detecting Drowsy Students?},
  author={Tim},
  school={University of Potsdam},
  year={2024},
  note={Cognitive Systems Master, Individual Research Project}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact & Support

For questions, issues, or suggestions:
- **GitHub Issues:** Open an issue on the repository
- **Email:** [Your contact email]

## Acknowledgments

- Dataset provided by [Source/Institution]
- Built with [TensorFlow](https://www.tensorflow.org/), [scikit-learn](https://scikit-learn.org/), and [Sacred](https://github.com/IDSIA/sacred)
- Research at University of Potsdam, Cognitive Systems Master program

---

**Last Updated:** February 2024  
**Version:** 1.0.0  
