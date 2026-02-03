"""Input validation utilities for drowsiness detection.

Provides validation decorators and utilities for function inputs and outputs.
"""

from typing import Callable, Any
import numpy as np
from functools import wraps

from drowsiness_detection.logging_utils import get_logger

logger = get_logger(__name__)


def validate_array_shape(expected_shape_fn: Callable) -> Callable:
    """Decorator to validate array shape before processing.
    
    Args:
        expected_shape_fn: Function that takes (actual_shape) and returns expected shape
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get first array argument
            for arg in args:
                if isinstance(arg, np.ndarray):
                    actual_shape = arg.shape
                    logger.debug(f"{func.__name__}: Input shape {actual_shape}")
                    break
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_num_targets(num_targets: int) -> None:
    """Validate that num_targets is supported.
    
    Args:
        num_targets: Number of target classes
        
    Raises:
        ValueError: If num_targets is not supported
    """
    supported = [2, 3, 5, 9]
    if num_targets not in supported:
        raise ValueError(
            f"num_targets={num_targets} is not supported. "
            f"Supported values: {supported}"
        )
    logger.debug(f"Validated num_targets={num_targets}")


def validate_threshold(threshold: float) -> None:
    """Validate threshold value.
    
    Args:
        threshold: Threshold value
        
    Raises:
        ValueError: If threshold is invalid
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    logger.debug(f"Validated threshold={threshold}")


def validate_path_exists(path) -> None:
    """Validate that a path exists.
    
    Args:
        path: Path object or string
        
    Raises:
        FileNotFoundError: If path doesn't exist
    """
    from pathlib import Path
    p = Path(path) if isinstance(path, str) else path
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    logger.debug(f"Validated path exists: {p}")
