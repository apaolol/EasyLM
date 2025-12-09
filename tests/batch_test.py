# ============================================================================
# FILE 4: tests/test_regression_stats_dunders.py
# ============================================================================
"""
Test dunder methods for RegressionStats
"""
import pytest
import numpy as np
from EasyLM import LinearModel

class TestRegressionStatsRepr:
    """Test __repr__ method."""
    
    def test_repr_shows_key_stats(self):
        """Test repr displays key statistics."""
        X = np.random.randn(60, 2)
        y = np.random.randn(60)
        
        model = LinearModel()
        model.fit(X, y)
        
        stats = model._stats
        repr_str = repr(stats)
        
        assert "RegressionStats" in repr_str
        assert "n_obs" in repr_str
        assert "n_params" in repr_str
        assert "R²" in repr_str or "R-squared" in repr_str


class TestRegressionStatsStr:
    """Test __str__ method."""
    
    def test_str_readable_summary(self):
        """Test str provides readable summary."""
        X = np.random.randn(40, 3)
        y = np.random.randn(40)
        
        model = LinearModel()
        model.fit(X, y)
        
        stats = model._stats
        str_output = str(stats)
        
        assert "Regression Statistics" in str_output
        assert "Observations" in str_output
        assert "Parameters" in str_output
        assert "R-squared" in str_output
        assert "AIC" in str_output


class TestRegressionStatsLen:
    """Test __len__ method."""
    
    def test_len_returns_n_obs(self):
        """Test len returns number of observations."""
        X = np.random.randn(85, 2)
        y = np.random.randn(85)
        
        model = LinearModel()
        model.fit(X, y)
        
        stats = model._stats
        assert len(stats) == 85


# ============================================================================
# FILE 5: tests/test_summary_formatter_dunders.py
# ============================================================================
"""
Test dunder methods for SummaryFormatter
"""
import pytest
import numpy as np
from EasyLM import LinearModel
from EasyLM.summary_formatter import SummaryFormatter

class TestSummaryFormatterRepr:
    """Test __repr__ method."""
    
    def test_repr_shows_format(self):
        """Test repr displays float format."""
        formatter = SummaryFormatter()
        repr_str = repr(formatter)
        
        assert "SummaryFormatter" in repr_str
        assert "float_format" in repr_str


class TestSummaryFormatterStr:
    """Test __str__ method."""
    
    def test_str_descriptive(self):
        """Test str provides description."""
        formatter = SummaryFormatter()
        str_output = str(formatter)
        
        assert "SummaryFormatter" in str_output


class TestSummaryFormatterCall:
    """Test __call__ method (makes formatter callable)."""
    
    def test_call_equivalent_to_format(self):
        """Test calling formatter same as format() method."""
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        
        model = LinearModel()
        model.fit(X, y)
        
        formatter = SummaryFormatter()
        coef_table = model._stats.get_coefficient_table()
        info = model._stats.get_summary_info()
        
        # Both should produce same output
        output1 = formatter.format(coef_table, info)
        output2 = formatter(coef_table, info)
        
        assert output1 == output2

