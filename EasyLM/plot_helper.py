"""
PlotHelper: plotting utilities for model comparison.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotHelper:
    """
    Visualization tools for comparing multiple models.
    Focuses on coefficient comparison and model fit visualization.
    """

    @staticmethod
    def plot_coefficients_comparison(models, labels=None, ax=None):
        """
        Compare coefficients across multiple models with a grouped bar chart.
        
        Parameters:
        -----------
        models : list
            List of fitted model objects with params_ attribute
        labels : list, optional
            Labels for each model. If None, uses "Model 1", "Model 2", etc.
        ax : matplotlib axes, optional
            Axes to plot on. If None, creates new figure.
            
        Returns:
        --------
        ax : matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if labels is None:
            labels = [f"Model {i+1}" for i in range(len(models))]
        
        # Get coefficients from all models
        max_params = max(len(m.params_) for m in models)
        coef_data = {}
        
        for i, (model, label) in enumerate(zip(models, labels)):
            coefs = model.params_
            # Pad with NaN if different lengths
            padded = np.pad(
                coefs,
                (0, max_params - len(coefs)),
                constant_values=np.nan
            )
            coef_data[label] = padded
        
        # Create grouped bar chart
        df = pd.DataFrame(coef_data)
        x = np.arange(len(df))
        width = 0.8 / len(models)
        
        for i, col in enumerate(df.columns):
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, df[col], width, label=col, alpha=0.8)
        
        ax.set_xlabel('Coefficient Index')
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Coefficient Comparison Across Models')
        ax.set_xticks(x)
        ax.set_xticklabels([f'β{i}' for i in range(len(df))])
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        return ax

    @staticmethod
    def plot_model_metrics(models, labels=None, ax=None):
        """
        Compare model fit metrics (AIC, BIC, R²) with a grouped bar chart.
        
        Parameters:
        -----------
        models : list
            List of fitted model objects with aic(), bic(), r_squared() methods
        labels : list, optional
            Labels for each model. If None, uses "Model 1", "Model 2", etc.
        ax : matplotlib axes, optional
            Axes to plot on. If None, creates new figure.
            
        Returns:
        --------
        ax : matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if labels is None:
            labels = [f"Model {i+1}" for i in range(len(models))]
        
        # Collect metrics
        metrics = {
            'AIC': [],
            'BIC': [],
            'R²': []
        }
        
        for model in models:
            metrics['AIC'].append(model.aic())
            metrics['BIC'].append(model.bic())
            metrics['R²'].append(model.r_squared())
        
        # Create grouped bar chart
        x = np.arange(len(labels))
        width = 0.25
        
        ax.bar(x - width, metrics['AIC'], width, label='AIC', alpha=0.8)
        ax.bar(x, metrics['BIC'], width, label='BIC', alpha=0.8)
        ax.bar(x + width, metrics['R²'], width, label='R²', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Metric Value')
        ax.set_title('Model Fit Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        return ax

    @staticmethod
    def plot_predictions_comparison(models, X, y, labels=None, ax=None):
        """
        Compare predictions from multiple models against observed values.
        
        Parameters:
        -----------
        models : list
            List of fitted model objects
        X : array-like
            Feature matrix for predictions
        y : array-like
            Observed values
        labels : list, optional
            Labels for each model. If None, uses "Model 1", "Model 2", etc.
        ax : matplotlib axes, optional
            Axes to plot on. If None, creates new figure.
            
        Returns:
        --------
        ax : matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if labels is None:
            labels = [f"Model {i+1}" for i in range(len(models))]
        
        # Plot observed values
        x_vals = np.arange(len(y))
        ax.scatter(
            x_vals,
            y,
            color='black',
            label='Observed',
            alpha=0.6,
            s=50,
            marker='o'
        )
        
        # Plot predictions from each model
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        for model, label, color in zip(models, labels, colors):
            predictions = model.predict(X)
            ax.plot(
                x_vals,
                predictions,
                label=label,
                alpha=0.7,
                linewidth=2,
                color=color
            )
        
        ax.set_xlabel('Observation Index')
        ax.set_ylabel('Value')
        ax.set_title('Model Predictions vs Observed Values')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return ax

    @staticmethod
    def plot_residuals_comparison(models, X, y, labels=None):
        """
        Compare residual patterns from multiple models in subplots.
        
        Parameters:
        -----------
        models : list
            List of fitted model objects
        X : array-like
            Feature matrix for predictions
        y : array-like
            Observed values
        labels : list, optional
            Labels for each model. If None, uses "Model 1", "Model 2", etc.
            
        Returns:
        --------
        fig : matplotlib figure
        """
        if labels is None:
            labels = [f"Model {i+1}" for i in range(len(models))]
        
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, model, label in zip(axes, models, labels):
            predictions = model.predict(X)
            residuals = y - predictions
            
            ax.scatter(predictions, residuals, alpha=0.6)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax.set_xlabel('Fitted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{label} Residuals')
            ax.grid(alpha=0.3)
        
        fig.suptitle(
            'Residual Comparison Across Models',
            fontsize=14,
            y=1.02
        )
        plt.tight_layout()
        
        return fig

    @staticmethod
    def plot_comprehensive_comparison(models, X, y, labels=None):
        """
        Create a comprehensive 2x2 comparison dashboard for models.
        
        Parameters:
        -----------
        models : list
            List of fitted model objects
        X : array-like
            Feature matrix for predictions
        y : array-like
            Observed values
        labels : list, optional
            Labels for each model. If None, uses "Model 1", "Model 2", etc.
            
        Returns:
        --------
        fig : matplotlib figure
        """
        if labels is None:
            labels = [f"Model {i+1}" for i in range(len(models))]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Coefficient comparison
        PlotHelper.plot_coefficients_comparison(models, labels, ax=axes[0, 0])
        
        # Metrics comparison
        PlotHelper.plot_model_metrics(models, labels, ax=axes[0, 1])
        
        # Predictions comparison
        PlotHelper.plot_predictions_comparison(
            models,
            X,
            y,
            labels,
            ax=axes[1, 0]
        )
        
        # R² comparison as horizontal bar
        r2_values = [m.r_squared() for m in models]
        axes[1, 1].barh(labels, r2_values, alpha=0.8)
        axes[1, 1].set_xlabel('R² Value')
        axes[1, 1].set_title('R² Comparison (Higher is Better)')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        fig.suptitle(
            'Comprehensive Model Comparison Dashboard',
            fontsize=16,
            y=0.995
        )
        plt.tight_layout()
        
        return fig
