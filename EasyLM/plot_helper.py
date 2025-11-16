"""
PlotHelper: plotting utilities for diagnostics.
"""

# pip-install hint:
# pip install matplotlib numpy pandas

import matplotlib.pyplot as plt
import numpy as np

class PlotHelper:
    """
    Simple plotting helpers. Keep matplotlib usage minimal and flexible.
    """

    @staticmethod
    def plot_fitted_vs_observed(fitted, observed, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(fitted, observed, alpha=0.6)
        mn = min(min(fitted), min(observed))
        mx = max(max(fitted), max(observed))
        ax.plot([mn, mx], [mn, mx], linestyle="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Observed values")
        ax.set_title("Fitted vs Observed")
        return ax

    @staticmethod
    def plot_residuals(fitted, residuals, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(fitted, residuals, alpha=0.6)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")
        return ax

    @staticmethod
    def plot_qq(residuals, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        import scipy.stats as stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q")
        return ax
