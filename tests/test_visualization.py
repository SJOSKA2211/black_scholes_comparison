"""
Test cases for Visualization Module
"""
import pytest
import numpy as np
import os
from visualization.plot_generator import (
    plot_convergence,
    plot_error_chart,
    plot_comparison
)

# Sample data for testing
@pytest.fixture
def sample_convergence_data():
    errors = [0.5, 0.2, 0.1, 0.05]
    grid_sizes = [100, 200, 400, 800]
    return errors, grid_sizes

@pytest.fixture
def sample_error_chart_data():
    method_names = ["Method A", "Method B", "Method C"]
    errors = [0.1, 0.05, 0.08]
    return method_names, errors

@pytest.fixture
def sample_comparison_data():
    actual_prices = [10, 20, 30, 40]
    predicted_prices = {
        "Method X": [10.1, 19.8, 30.5, 39.7],
        "Method Y": [9.9, 20.2, 29.8, 40.3]
    }
    return actual_prices, predicted_prices

def test_plot_convergence_display(sample_convergence_data):
    errors, grid_sizes = sample_convergence_data
    # Test that the function runs without error when displaying
    plot_convergence(errors, grid_sizes, title="Test Convergence Plot")
    assert True # If no exception, test passes

def test_plot_convergence_save(sample_convergence_data):
    errors, grid_sizes = sample_convergence_data
    filename = "test_convergence_plot.png"
    plot_convergence(errors, grid_sizes, title="Test Convergence Plot", filename=filename)
    assert os.path.exists(filename)
    os.remove(filename) # Clean up

def test_plot_error_chart_display(sample_error_chart_data):
    method_names, errors = sample_error_chart_data
    # Test that the function runs without error when displaying
    plot_error_chart(method_names, errors, title="Test Error Chart")
    assert True # If no exception, test passes

def test_plot_error_chart_save(sample_error_chart_data):
    method_names, errors = sample_error_chart_data
    filename = "test_error_chart.png"
    plot_error_chart(method_names, errors, title="Test Error Chart", filename=filename)
    assert os.path.exists(filename)
    os.remove(filename) # Clean up

def test_plot_comparison_display(sample_comparison_data):
    actual_prices, predicted_prices = sample_comparison_data
    # Test that the function runs without error when displaying
    plot_comparison(actual_prices, predicted_prices, title="Test Comparison Plot")
    assert True # If no exception, test passes

def test_plot_comparison_save(sample_comparison_data):
    actual_prices, predicted_prices = sample_comparison_data
    filename = "test_comparison_plot.png"
    plot_comparison(actual_prices, predicted_prices, title="Test Comparison Plot", filename=filename)
    assert os.path.exists(filename)
    os.remove(filename) # Clean up
