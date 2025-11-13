"""
Statistical Testing Framework
Provides functions for various statistical analyses of pricing method results.
"""
import numpy as np
from scipy import stats
from typing import Tuple, List

def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculates the confidence interval for a given set of data.
    
    Parameters:
    -----------
    data : List[float]
        The input data.
    confidence : float, optional
        The confidence level (e.g., 0.95 for 95% confidence interval).
        Defaults to 0.95.
        
    Returns:
    --------
    Tuple[float, float]
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m - h, m + h

def paired_t_test(data1: List[float], data2: List[float]) -> Tuple[float, float]:
    """
    Performs a paired (dependent) t-test on two related samples.
    
    Parameters:
    -----------
    data1 : List[float]
        The first set of data.
    data2 : List[float]
        The second set of data.
        
    Returns:
    --------
    Tuple[float, float]
        A tuple containing the t-statistic and the two-sided p-value.
    """
    return stats.ttest_rel(data1, data2)

def wilcoxon_signed_rank_test(data1: List[float], data2: List[float]) -> Tuple[float, float]:
    """
    Performs the Wilcoxon signed-rank test on two related samples.
    
    Parameters:
    -----------
    data1 : List[float]
        The first set of data.
    data2 : List[float]
        The second set of data.
        
    Returns:
    --------
    Tuple[float, float]
        A tuple containing the Wilcoxon statistic and the two-sided p-value.
    """
    return stats.wilcoxon(data1, data2)

def correlation_analysis(data1: List[float], data2: List[float]) -> float:
    """
    Calculates the Pearson correlation coefficient between two datasets.
    
    Parameters:
    -----------
    data1 : List[float]
        The first set of data.
    data2 : List[float]
        The second set of data.
        
    Returns:
    --------
    float
        The Pearson correlation coefficient.
    """
    return np.corrcoef(data1, data2)[0, 1]

# Placeholders for ANOVA, Tukey HSD, Effect Size (Cohen's d)
# These often require more complex data structures or specific use cases.
