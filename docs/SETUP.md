# Setup and Installation Guide

This guide walks you through setting up the drowsiness-detection project on your local machine.

## Prerequisites

- **Python:** 3.8 or higher
- **pip:** Package installer for Python
- **git:** Version control (for cloning the repository)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/tpatzelt/drowsiness-detection.git
cd drowsiness-detection
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to isolate project dependencies.

**Using venv (built-in):**
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**Using conda (if you have Anaconda installed):**
```bash
conda create -n drowsiness python=3.9
conda activate drowsiness
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For development (includes testing and linting tools):
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 4. Configure Data Path

The project requires access to preprocessed drowsiness data.

**Option A: Set environment variable (recommended)**
```bash
export DROWSINESS_DATA_PATH="/path/to/your/drowsiness_data"
```

**Option B: Edit config directly**
Edit `drowsiness_detection/config.py`:
```python
from pathlib import Path
BASE_PATH = Path("/your/data/path")
```

### 5. Verify Installation

```bash
# Test Python import
python -c "import drowsiness_detection; print('Success!')"

# Run tests (if pytest is installed)
pytest tests/ -v
```

## Data Setup

### Dataset Structure

Your data should be organized as follows:

```
drowsiness_data/
└── Windows_1_Hz/
    ├── WindowData/
    │   ├── 1_sec/
    │   │   └── *.npy files with eye-tracking signals
    │   └── Format/
    │       └── data_format.json
    ├── WindowLabels/
    │   └── KSS/
    │       └── 1_sec/
    │           └── *.npy files with labels
    ├── WindowFeatures/
    │   ├── 1_sec/
    │   │   └── *.npy files with engineered features
    │   └── feature_names.txt
    └── TrainTestSplits/
        └── 1_sec/
            └── Train/test split files
```

### Preprocessing Raw Data

If you have raw data that needs preprocessing:

1. **Open the preprocessing notebook:**
   ```bash
   jupyter notebook notebooks/create_preprocessed_data.ipynb
   ```

2. **Set the input path** in the first cell to your raw data location

3. **Run all cells** to generate preprocessed data

4. **Output** will be saved to `data/preprocessed/`

## Development Setup

### Install Development Dependencies

```bash
pip install -e ".[dev,notebooks]"
```

This installs:
- pytest and coverage tools
- Code formatting (black, isort)
- Linting (flake8, mypy)
- Jupyter notebooks

### Run Tests

```bash
pytest tests/ -v --cov=drowsiness_detection
```

### Format Code

```bash
black drowsiness_detection/
isort drowsiness_detection/
```

### Check Code Quality

```bash
flake8 drowsiness_detection/
mypy drowsiness_detection/ --ignore-missing-imports
```

## Running Experiments

### Quick Start

```bash
python drowsiness_detection/run_grid_search_experiment.py \
    --config_file drowsiness_detection/config_example.json \
    --seed 42
```

### Using Sacred

The experiment runner uses Sacred for experiment tracking:

```bash
python drowsiness_detection/run_grid_search_experiment.py \
    with model='random_forest' \
    data_params.frequency=1 \
    data_params.seconds=1
```

### Results

Experiment results are saved to `logs/` with:
- `config.json` - Experiment configuration
- `metrics.json` - Evaluation metrics
- `train_history.csv` - Training history (for neural networks)
- `best_model/` - Saved model files

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'drowsiness_detection'`

**Solution:** Make sure you're in the correct directory and virtual environment is activated:
```bash
cd drowsiness-detection
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### Data Path Not Found

**Problem:** `FileNotFoundError: Data directory not found`

**Solution:** Set the correct data path:
```bash
export DROWSINESS_DATA_PATH="/correct/path"
# Or edit config.py directly
```

### TensorFlow/CUDA Issues

**Problem:** TensorFlow can't find CUDA or GPU

**Solution:** TensorFlow will fall back to CPU automatically. For GPU support, install CUDA-compatible versions:
```bash
pip install tensorflow-gpu  # If you have NVIDIA GPU
```

### Memory Issues

**Problem:** Out of memory errors when loading data

**Solution:** 
- Reduce batch size in experiment config
- Process data in smaller chunks
- Use a machine with more RAM

### Virtual Environment Not Activating

**Problem:** Commands not found after activating venv

**Solution:** Make sure the activation script is in the correct location:
```bash
# Linux/macOS
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (Command Prompt)
venv\Scripts\activate.bat
```

## Next Steps

- [API Documentation](API.md) - Detailed API reference
- [README](../README.md) - Project overview and usage examples
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- Check `notebooks/` for example analysis and experiments

## Getting Help

- Review the example scripts in `notebooks/`
- Check the [README](../README.md) for common usage patterns
- Open an issue on GitHub for bugs or questions
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines
