from functools import reduce
from pathlib import Path

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import pandas as pd


def print_nan_intersections(df: pd.DataFrame) -> None:
    """Print analysis of NaN values across DataFrame columns.
    
    Shows which NaN values in each column are unique to that column (not appearing
    in any other column).
    
    Args:
        df: DataFrame to analyze
    """
    nan_indices = dict()
    for column in df.columns:
        nan_indices[column] = set(df.loc[df[column].isna()].index)

    for column in df.columns:
        nan_indices_copy = nan_indices.copy()
        col_indices = set(nan_indices_copy.pop(column))
        other_indices = reduce(lambda x, y: set(x) | set(y), nan_indices_copy.values())
        unique_nans = col_indices - other_indices
        print(f"Column {column} has {len(unique_nans)} nans that appear in no other column.")


def digitize(arr: np.ndarray, shift: float = 0.5, label_start_index: int = 1) -> np.ndarray:
    """Digitize array values into discrete bins.
    
    Args:
        arr: Input array to digitize
        shift: Shift for bin edges (default 0.5)
        label_start_index: Starting index for labels (default 1)
        
    Returns:
        Digitized array with labels
    """
    bins = [x + shift for x in range(1, 9)]
    return np.digitize(arr, bins=bins) + label_start_index


def label_to_one_hot_like(arr: np.ndarray, k: int = 9) -> np.ndarray:
    """Convert ordinal labels to one-hot-like encoding for ordinal regression.
    
    Creates a binary matrix where row i has 1s up to index arr[i]-1, representing
    the ordinal relationship between classes.
    
    Args:
        arr: Ordinal labels (typically 1-9)
        k: Maximum label value + 1
        
    Returns:
        Binary matrix of shape (len(arr), k-1)
    """
    arr = np.squeeze(arr)
    one_hots = np.zeros(shape=(len(arr), k - 1), dtype=int)
    for i, element in enumerate(arr):
        one_hots[i, :element] = 1
    return one_hots


def binarize(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Convert continuous values to binary labels based on threshold.
    
    Args:
        arr: Input array
        threshold: Threshold value for binarization
        
    Returns:
        Binary array (1 if >= threshold, 0 otherwise)
    """
    return (arr >= threshold).astype(int)


class ArrayWrapper:
    """Context manager for writing large arrays to disk in chunks.
    
    Useful for processing very large datasets that don't fit in memory.
    
    Attributes:
        max_size_mb: Maximum size of each chunk in MB
        directory: Directory to save chunks
        name_gen: Generator for chunk filenames
        array: Current chunk array
        n_cols: Number of columns
    """
    def __init__(self, directory: Path, filename_generator, max_size_mb: int = 1, n_cols: int = 1):
        """Initialize ArrayWrapper.
        
        Args:
            directory: Path to save chunk files
            filename_generator: Generator yielding filenames
            max_size_mb: Maximum size per chunk in MB
            n_cols: Number of columns in array
        """
        self.max_size_mb = max_size_mb
        self.directory = directory
        self.name_gen = filename_generator
        self.array = create_emtpy_array_of_max_size(max_size_mb=max_size_mb, n_cols=n_cols)
        self.i = iter(range(self.array.shape[0]))
        self.n_cols = n_cols

    def add(self, row: np.ndarray) -> None:
        """Add a row to the current chunk, saving when full.
        
        Args:
            row: Row to add
        """
        try:
            next_i = next(self.i)
            self.array[next_i] = row
        except StopIteration:
            np.save(file=self.directory.joinpath(next(self.name_gen)), arr=self.array)
            self.array = create_emtpy_array_of_max_size(max_size_mb=self.max_size_mb,
                                                        n_cols=self.n_cols)
            self.i = iter(range(self.array.shape[0]))

    def close(self) -> None:
        """Save final chunk and cleanup."""
        np.save(file=self.directory.joinpath(next(self.name_gen)), arr=self.array[:next(self.i)])


def create_emtpy_array_of_max_size(max_size_mb: int, n_cols: int) -> np.ndarray:
    """Create empty array with maximum size in MB.
    
    Args:
        max_size_mb: Maximum size in MB
        n_cols: Number of columns
        
    Returns:
        Empty float array
    """
    max_bytes = MB_to_bytes(mb=max_size_mb)
    max_rows = np.floor(max_bytes / 8 / n_cols)
    return np.empty(shape=(max_rows.astype(int), n_cols))


def bytes_to_MB(b: int) -> float:
    """Convert bytes to megabytes.
    
    Args:
        b: Size in bytes
        
    Returns:
        Size in MB
    """
    return b / 1024 / 1024


def MB_to_bytes(mb: int) -> int:
    """Convert megabytes to bytes.
    
    Args:
        mb: Size in MB
        
    Returns:
        Size in bytes
    """
    return mb * 1024 * 1024


def name_generator(base_name: str):
    """Generate sequential filenames.
    
    Args:
        base_name: Base name for files
        
    Yields:
        Filenames: base_name_0, base_name_1, etc.
    """
    i = 0
    while True:
        yield base_name + "_" + str(i)
        i += 1


def spec_to_config_space(specs: list) -> CS.ConfigurationSpace:
    """Convert specification list to ConfigSpace.
    
    Specs should be dicts with 'name' and 'kwargs' keys, where name is a
    ConfigSpace hyperparameter class name.
    
    Args:
        specs: List of hyperparameter specifications
        
    Returns:
        ConfigSpace ConfigurationSpace object
        
    Example:
        specs = [
            {"name": "UniformIntegerHyperparameter", "kwargs": {"name": "max_depth", "lower": 1, "upper": 10}},
            {"name": "UniformIntegerHyperparameter", "kwargs": {"name": "min_samples", "lower": 1, "upper": 20}}
        ]
    """
    cs = CS.ConfigurationSpace()
    for spec in specs:
        to_add = getattr(CSH, spec["name"])
        cs.add_hyperparameter(to_add(**spec["kwargs"]))
    return cs

