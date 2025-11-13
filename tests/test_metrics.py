"""
Test cases for Performance Metrics Module
"""
import pytest
import numpy as np
from analysis.performance_metrics import (
    absolute_error,
    relative_error,
    mean_absolute_percentage_error,
    root_mean_square_error,
    execution_time_profiler
)
import time

def test_absolute_error():
    assert absolute_error(10, 8) == 2
    assert absolute_error(8, 10) == 2
    assert absolute_error(5, 5) == 0
    assert absolute_error(0, 0) == 0
    assert absolute_error(-5, -7) == 2

def test_relative_error():
    assert relative_error(10, 8) == 20.0
    assert relative_error(8, 10) == 25.0
    assert relative_error(5, 5) == 0.0
    assert relative_error(0, 0) == 0.0 # Should handle division by zero
    assert relative_error(100, 90) == 10.0
    assert relative_error(100, 110) == 10.0

def test_mean_absolute_percentage_error():
    actual = np.array([10, 20, 30])
    predicted = np.array([12, 18, 33])
    # abs((10-12)/10) * 100 = 20
    # abs((20-18)/20) * 100 = 10
    # abs((30-33)/30) * 100 = 10
    # (20 + 10 + 10) / 3 = 13.333...
    assert np.isclose(mean_absolute_percentage_error(actual, predicted), 13.333333333333334)

    actual_zero = np.array([0, 10, 20])
    predicted_zero = np.array([0, 12, 18])
    # Only non-zero actual values are considered
    # abs((10-12)/10) * 100 = 20
    # abs((20-18)/20) * 100 = 10
    # (20 + 10) / 2 = 15
    assert np.isclose(mean_absolute_percentage_error(actual_zero, predicted_zero), 15.0)

    all_zero_actual = np.array([0, 0, 0])
    all_zero_predicted = np.array([0, 0, 0])
    assert mean_absolute_percentage_error(all_zero_actual, all_zero_predicted) == 0.0

def test_root_mean_square_error():
    actual = np.array([10, 20, 30])
    predicted = np.array([12, 18, 33])
    # ((10-12)^2 + (20-18)^2 + (30-33)^2) / 3
    # (4 + 4 + 9) / 3 = 17 / 3 = 5.666...
    # sqrt(5.666...) = 2.380476...
    assert np.isclose(root_mean_square_error(actual, predicted), 2.380476142847005)

    actual_single = np.array([10])
    predicted_single = np.array([12])
    assert np.isclose(root_mean_square_error(actual_single, predicted_single), 2.0)

def test_execution_time_profiler():
    @execution_time_profiler
    def dummy_function(delay_time):
        time.sleep(delay_time)
        return "done"

    result, elapsed_time = dummy_function(0.1)
    assert result == "done"
    assert elapsed_time >= 0.1
    assert elapsed_time < 0.2 # Should not take excessively long
