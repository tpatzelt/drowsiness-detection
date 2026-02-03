"""Drowsiness detection package for comparing engineered features vs. neural networks."""

__version__ = "1.0.0"
__author__ = "Tim"

from drowsiness_detection.config import set_paths, Paths
from drowsiness_detection.logging_utils import get_logger, set_log_level
from drowsiness_detection import constants

__all__ = [
    "set_paths",
    "Paths",
    "get_logger",
    "set_log_level",
    "constants",
]
