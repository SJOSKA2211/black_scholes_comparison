"""
Visualization Module
Provides functions for generating various plots and charts for analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

def plot_convergence(errors: List[float], grid_sizes: List[int], title: str = "Convergence Plot", filename: str = None):
    """
    Generates a convergence plot (error vs. grid size).
    
    Parameters:
    -----------
    errors : List[float]
        List of errors corresponding to different grid sizes.
    grid_sizes : List[int]
        List of grid sizes (e.g., number of steps, M or N).
    title : str
        Title of the plot.
    filename : str, optional
        If provided, saves the plot to this filename.
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(grid_sizes, errors, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Grid Size (log scale)")
    plt.ylabel("Error (log scale)")
    plt.grid(True, which="both", ls="-")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_error_chart(method_names: List[str], errors: List[float], title: str = "Error Comparison", filename: str = None):
    """
    Generates a bar chart comparing errors across different methods.
    
    Parameters:
    -----------
    method_names : List[str]
        Names of the pricing methods.
    errors : List[float]
        Errors corresponding to each method.
    title : str
        Title of the plot.
    filename : str, optional
        If provided, saves the plot to this filename.
    """
    plt.figure(figsize=(12, 7))
    plt.bar(method_names, errors, color='skyblue')
    plt.title(title)
    plt.xlabel("Pricing Method")
    plt.ylabel("Error")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_comparison(actual_prices: List[float], predicted_prices: Dict[str, List[float]], title: str = "Actual vs. Predicted Prices", filename: str = None):
    """
    Generates a scatter plot comparing actual prices against predicted prices from various methods.
    
    Parameters:
    -----------
    actual_prices : List[float]
        List of actual (e.g., analytical or market) prices.
    predicted_prices : Dict[str, List[float]]
        Dictionary where keys are method names and values are lists of predicted prices.
    title : str
        Title of the plot.
    filename : str, optional
        If provided, saves the plot to this filename.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(actual_prices, actual_prices, color='red', label='Actual (Reference)', marker='x') # Reference line
    
    for method_name, prices in predicted_prices.items():
        plt.scatter(actual_prices, prices, label=method_name, alpha=0.7)
        
    plt.title(title)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

# Placeholder for other visualizations like 3D option surface plots, Greeks visualization, etc.
# These would require more complex data structures and potentially more specialized libraries.
