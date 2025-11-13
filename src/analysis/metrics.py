"""
Performance Metrics Module
Provides functions to calculate various performance metrics for option pricing methods.
"""
import numpy as np
import time
from typing import Callable, Any, Tuple

def absolute_error(actual: float, predicted: float) -> float:
    """
    Calculates the absolute error between actual and predicted values.
    """
    return abs(actual - predicted)

def relative_error(actual: float, predicted: float) -> float:
    """
    Calculates the relative percentage error between actual and predicted values.
    Returns 0 if actual is 0 to avoid division by zero.
    """
    if actual == 0:
        return 0.0
    return absolute_error(actual, predicted) / actual * 100

def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Percentage Error (MAPE).
    Handles cases where actual values are zero.
    """
    if not isinstance(actual, np.ndarray):
        actual = np.array(actual)
    if not isinstance(predicted, np.ndarray):
        predicted = np.array(predicted)

    # Avoid division by zero for actual values that are zero
    # Replace 0 actual values with a small epsilon or handle them separately
    # For now, we'll exclude them from the calculation as is common.
    non_zero_actual_indices = actual != 0
    if not np.any(non_zero_actual_indices):
        return 0.0 # All actual values are zero, MAPE is undefined or 0

    return np.mean(np.abs((actual[non_zero_actual_indices] - predicted[non_zero_actual_indices]) / actual[non_zero_actual_indices])) * 100

def root_mean_square_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Error (RMSE).
    """
    if not isinstance(actual, np.ndarray):
        actual = np.array(actual)
    if not isinstance(predicted, np.ndarray):
        predicted = np.array(predicted)
    return np.sqrt(np.mean((actual - predicted)**2))

def execution_time_profiler(func: Callable) -> Callable:
    """
    A decorator to measure the execution time of a function.
    The decorated function will return its original result and the execution time.
    """
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, (end_time - start_time)
    return wrapper

# Placeholder for memory usage tracker (requires external libraries like 'psutil')
# For now, we'll just define a function that returns 0 or raises NotImplementedError
def memory_usage_tracker() -> float:
    """
    Tracks memory usage of the current process.
    Requires 'psutil' library. Placeholder for now.
    """
    # import psutil
    # process = psutil.Process()
    # return process.memory_info().rss / (1024 * 1024) # Memory in MB
    return 0.0 # Placeholder
    # raise NotImplementedError("Memory usage tracking requires 'psutil' library.")

# Placeholder for convergence rate estimator and stability indicator
# These would typically be part of a larger analysis framework, not standalone functions.
# They depend on running multiple simulations with varying parameters.
