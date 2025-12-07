"""
PlotHelper: plotting utilities for comparing regression models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Optional


class PlotHelper:
    """
    Simple plotting toolkit for comparing multiple regression models.
    
    Main features:
    - Compare coefficients across models
    - Compare fit metrics (AIC, BIC, R²)
    - Visualize predictions vs observed values
    - Check residual patterns
    """
    
    # HELPER METHODS
    
    @staticmethod
    def _get_model_name(model, index):
        """Get model name, or generate one if not available."""
        return getattr(model, 'name', f'Model {index+1}')
    
    @staticmethod
    def _extract_coefficients(model):
        """Get coefficient array from model."""
        if hasattr(model, 'params_'):
            return np.asarray(model.params_)
        if hasattr(model, '_stats') and model._stats is not None:
            return np.asarray(model._stats.coefficients)
        return np.array([np.nan])
    
    @staticmethod
    def _get_metric_value(model, metric_name):
        """Get a metric value (aic, bic, r_squared) from model."""
        try:
            if hasattr(model, metric_name):
                attr = getattr(model, metric_name)
                if callable(attr):
                    return float(attr())
                return float(attr)
            return np.nan
        except Exception:
            return np.nan
    
    # PLOTTING METHODS
    
    @staticmethod
    def plot_coefficients_comparison(models, labels=None, ax=None, figsize=(10, 6), **kwargs):
        """Compare coefficients across models with bar chart."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if labels is None:
            labels = [PlotHelper._get_model_name(m, i) for i, m in enumerate(models)]
        
        # Extract coefficients and pad to same length
        all_coefs = [PlotHelper._extract_coefficients(m) for m in models]
        max_len = max(len(c) for c in all_coefs)
        
        padded_coefs = {}
        for label, coefs in zip(labels, all_coefs):
            padded = np.full(max_len, np.nan)
            padded[:len(coefs)] = coefs
            padded_coefs[label] = padded
        
        # Create bar chart
        df = pd.DataFrame(padded_coefs)
        x_positions = np.arange(max_len)
        bar_width = 0.8 / len(models)
        
        # Build bar_kwargs directly from kwargs, with defaults
        bar_kwargs = {'alpha': kwargs.get('alpha', 0.8)}
        if 'color' in kwargs:
            bar_kwargs['color'] = kwargs['color']
        if 'edgecolor' in kwargs:
            bar_kwargs['edgecolor'] = kwargs['edgecolor']
        
        for i, column in enumerate(df.columns):
            offset = (i - len(models)/2 + 0.5) * bar_width
            ax.bar(x_positions + offset, df[column], bar_width, label=column, **bar_kwargs)
        
        ax.set_xlabel('Coefficient Index')
        ax.set_ylabel('Value')
        ax.set_title('Coefficient Comparison')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'β{i}' for i in range(max_len)])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_model_metrics(models, labels=None, ax=None, figsize=(10, 6), **kwargs):
        """Compare AIC, BIC, and R² across models."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if labels is None:
            labels = [PlotHelper._get_model_name(m, i) for i, m in enumerate(models)]
        
        # Extract metrics
        aic_values = [PlotHelper._get_metric_value(m, 'aic') for m in models]
        bic_values = [PlotHelper._get_metric_value(m, 'bic') for m in models]
        r2_values = [PlotHelper._get_metric_value(m, 'r_squared') for m in models]
        
        # Build bar_kwargs directly from kwargs, with defaults
        bar_kwargs = {'alpha': kwargs.get('alpha', 0.8)}
        if 'edgecolor' in kwargs:
            bar_kwargs['edgecolor'] = kwargs['edgecolor']
        
        # Create grouped bar chart
        x_positions = np.arange(len(labels))
        width = 0.25
        
        ax.bar(x_positions - width, aic_values, width, label='AIC', **bar_kwargs)
        ax.bar(x_positions, bic_values, width, label='BIC', **bar_kwargs)
        ax.bar(x_positions + width, r2_values, width, label='R²', **bar_kwargs)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Metric Value')
        ax.set_title('Model Fit Metrics')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_predictions_comparison(models, X, y, labels=None, ax=None, figsize=(10, 6), **kwargs):
        """Compare model predictions against observed values."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if labels is None:
            labels = [PlotHelper._get_model_name(m, i) for i, m in enumerate(models)]
        
        # Plot observed values
        x_vals = np.arange(len(y))
        ax.scatter(x_vals, y, color='black', label='Observed', alpha=0.6, s=50)
        
        # Build plot_kwargs directly from kwargs, with defaults
        plot_kwargs = {
            'alpha': kwargs.get('alpha', 0.7),
            'linewidth': kwargs.get('linewidth', 2),
            'linestyle': kwargs.get('linestyle', '-')
        }
        if 'marker' in kwargs:
            plot_kwargs['marker'] = kwargs['marker']
        
        # Plot predictions from each model
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        for model, label, color in zip(models, labels, colors):
            try:
                predictions = model.predict(X).ravel()
                ax.plot(x_vals, predictions, label=label, color=color, **plot_kwargs)
            except Exception as e:
                print(f"Warning: Could not plot {label}: {e}")
                continue
        
        ax.set_xlabel('Observation Index')
        ax.set_ylabel('Value')
        ax.set_title('Predictions vs Observed')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_residuals_comparison(models, X, y, labels=None, figsize=None, **kwargs):
        """Compare residual patterns from multiple models."""
        # Prepare data
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if labels is None:
            labels = [PlotHelper._get_model_name(m, i) for i, m in enumerate(models)]
        
        n_models = len(models)
        if figsize is None:
            figsize = (6 * n_models, 5)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        # Handle single model case
        if n_models == 1:
            axes = [axes]
        
        # Build scatter_kwargs directly from kwargs, with defaults
        scatter_kwargs = {
            'alpha': kwargs.get('alpha', 0.6),
        }
        if 'color' in kwargs:
            scatter_kwargs['color'] = kwargs['color']
        if 's' in kwargs:
            scatter_kwargs['s'] = kwargs['s']
        
        for ax, model, label in zip(axes, models, labels):
            try:
                predictions = model.predict(X).ravel()
                residuals = y - predictions
                
                ax.scatter(predictions, residuals, **scatter_kwargs)
                ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
                ax.set_xlabel('Fitted Values')
                ax.set_ylabel('Residuals')
                ax.set_title(f'{label}')
                ax.grid(alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        fig.suptitle('Residual Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_comprehensive_comparison(models, X, y, labels=None, figsize=(14, 10)):
        """Create a 2x2 dashboard with all comparisons."""
        if labels is None:
            labels = [PlotHelper._get_model_name(m, i) for i, m in enumerate(models)]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Coefficients
        PlotHelper.plot_coefficients_comparison(models, labels, ax=axes[0, 0])
        
        # 2. Metrics
        PlotHelper.plot_model_metrics(models, labels, ax=axes[0, 1])
        
        # 3. Predictions
        PlotHelper.plot_predictions_comparison(models, X, y, labels, ax=axes[1, 0])
        
        # 4. R² comparison
        r2_values = [PlotHelper._get_metric_value(m, 'r_squared') for m in models]
        axes[1, 1].barh(labels, r2_values, alpha=0.8)
        axes[1, 1].set_xlabel('R² Value')
        axes[1, 1].set_title('R² Comparison')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        fig.suptitle('Model Comparison Dashboard', fontsize=16, y=0.995)
        plt.tight_layout()
        
        return fig