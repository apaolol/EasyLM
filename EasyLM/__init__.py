"""
EasyLM package init
"""

from .base_model import BaseModel
from .linear_model import LinearModel
from .model_comparator import ModelComparator
from .plot_helper import PlotHelper
from .summary_formatter import SummaryFormatter
from .exceptions import EasyLMError, FitError, PredictError
from .utils import add_constant, check_array

__all__ = [
    "BaseModel",
    "LinearModel",
    "ModelComparator",
    "PlotHelper",
    "SummaryFormatter",
    "EasyLMError",
    "FitError",
    "PredictError",
    "add_constant",
    "check_array",
]