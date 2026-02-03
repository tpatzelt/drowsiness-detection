"""Configuration constants for drowsiness detection.

This module contains all magic numbers, thresholds, and configuration constants
to make the codebase more maintainable and configurable.
"""

# Label classification constants
KSS_THRESHOLD = 7  # Karolinska Sleepiness Scale threshold for binary classification
KSS_DISCRETIZATION_BINS = {
    2: [7],  # Binary: drowsy (>=7) vs not drowsy (<7)
    3: [6, 8],  # Tertiary: alert (<6), neutral (6-8), drowsy (>8)
    5: [3, 5, 7, 9],  # Quintile
    9: list(range(1, 9))  # Full KSS scale (1-9)
}

# Feature indices for engineered features
BLINK_RELATED_FEATURES = (5, 8, 9, 14, 15, 16, 19)
"""Indices of eye blink related features in feature vector."""

SACCADE_RELATED_FEATURES = (20, 21, 22)
"""Indices of saccade related features in feature vector."""

PUPIL_RELATED_FEATURES = (10, 11, 12, 13)
"""Indices of pupil related features in feature vector."""

# Neural network defaults
DEFAULT_LSTM_UNITS = 128
DEFAULT_LSTM_DROPOUT = 0.2
DEFAULT_CNN_FILTERS = 32
DEFAULT_CNN_KERNEL_SIZE = 5
DEFAULT_CNN_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 0.001

# Data processing constants
INTERPOLATED_KSS_INDEX = 2  # Index of interpolated KSS values in label files
SESSION_TYPE_MAPPING = {'a': 1, 'b': 2, 'e': 3, 's': 4}
"""Mapping of session type letters to integer codes."""

# File I/O constants
DEFAULT_MAX_FILE_SIZE_MB = 100
DEFAULT_FEATURE_COLS = 786  # Number of feature columns including label
DEFAULT_TRAIN_FILES = 2
DEFAULT_TEST_FILES = 1

# Labels for different number of classes
LABEL_NAMES = {
    2: ["not drowsy", "drowsy"],
    3: ["active", "neutral", "drowsy"],
    5: ["alert", "awake", "neutral", "slightly drowsy", "drowsy"],
    9: [str(num) for num in range(1, 10)]
}
