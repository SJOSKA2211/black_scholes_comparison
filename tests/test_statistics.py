"""
Test cases for Statistical Testing Framework
"""
import pytest
import numpy as np
from scipy import stats
from analysis.statistical_tests import (
    calculate_confidence_interval,
    paired_t_test,
    wilcoxon_signed_rank_test,
    correlation_analysis
)

# Sample data for testing
@pytest.fixture
def sample_data_ci():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

@pytest.fixture
def sample_data_paired_t():
    data1 = [10, 12, 15, 11, 13]
    data2 = [9, 11, 14, 10, 12]
    return data1, data2

@pytest.fixture
def sample_data_wilcoxon():
    data1 = [10, 12, 15, 11, 13]
    data2 = [9, 11, 14, 10, 12]
    return data1, data2

@pytest.fixture
def sample_data_correlation():
    data1 = [1, 2, 3, 4, 5]
    data2 = [2, 4, 5, 4, 5]
    return data1, data2

def test_calculate_confidence_interval(sample_data_ci):
    lower, upper = calculate_confidence_interval(sample_data_ci, confidence=0.95)
    # Expected values can be calculated using scipy.stats.t.interval
    mean = np.mean(sample_data_ci)
    sem = stats.sem(sample_data_ci)
    df = len(sample_data_ci) - 1
    interval = stats.t.interval(0.95, df, loc=mean, scale=sem)
    assert np.isclose(lower, interval[0])
    assert np.isclose(upper, interval[1])

def test_paired_t_test(sample_data_paired_t):
    data1, data2 = sample_data_paired_t
    t_stat, p_value = paired_t_test(data1, data2)
    expected_t_stat, expected_p_value = stats.ttest_rel(data1, data2)
    assert np.isclose(t_stat, expected_t_stat)
    assert np.isclose(p_value, expected_p_value)

def test_wilcoxon_signed_rank_test(sample_data_wilcoxon):
    data1, data2 = sample_data_wilcoxon
    wilcoxon_stat, p_value = wilcoxon_signed_rank_test(data1, data2)
    expected_wilcoxon_stat, expected_p_value = stats.wilcoxon(data1, data2)
    assert np.isclose(wilcoxon_stat, expected_wilcoxon_stat)
    assert np.isclose(p_value, expected_p_value)

def test_correlation_analysis(sample_data_correlation):
    data1, data2 = sample_data_correlation
    correlation = correlation_analysis(data1, data2)
    expected_correlation = np.corrcoef(data1, data2)[0, 1]
    assert np.isclose(correlation, expected_correlation)
