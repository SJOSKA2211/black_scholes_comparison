"""
Utility decorators for performance measurement.
"""
import time
import functools
import os
import psutil

def time_it(func):
    """
    A decorator that measures the execution time of a function.
    The decorated function will return its original result and the execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # If the function already returns a dict, add time to it.
        # Otherwise, return a new dict with result and time.
        if isinstance(result, dict):
            result['computation_time'] = elapsed_time
            return result
        return {"result": result, "computation_time": elapsed_time}
    return wrapper

def measure_memory(func):
    """
    A decorator that measures the peak memory usage of a function.
    Requires the 'psutil' library.
    The decorated function will return its original result and the peak memory usage in MB.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss # in bytes
        
        result = func(*args, **kwargs)
        
        # Get final memory usage
        final_memory = process.memory_info().rss # in bytes
        
        # Calculate peak memory usage during execution (this is a simplification,
        # true peak requires more advanced profiling or sampling)
        # For simplicity, we'll report the difference from initial to final,
        # or just the final if that's what's desired.
        # For a more accurate "peak", one would need to sample memory during execution.
        peak_memory_mb = (final_memory - initial_memory) / (1024 * 1024) # Convert to MB
        
        if isinstance(result, dict):
            result['peak_memory_mb'] = peak_memory_mb
            return result
        return {"result": result, "peak_memory_mb": peak_memory_mb}
    return wrapper
