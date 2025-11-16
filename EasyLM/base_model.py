# In EasyLM/base_model.py

from abc import ABC, abstractmethod
import pandas as pd
from .exceptions import ModelNotFittedError

class BaseModel(ABC):
    """
    Abstract Base Class for all models in the EasyLM library.

    This class defines the standard API for fitting, predicting,
    and summarizing models.

    Parameters:
    data : pd.DataFrame
        The DataFrame containing the data.
    formula : str
        An R-style formula string (e.g., "y ~ x1 + x2").
    """
    
    def __init__(self, data: pd.DataFrame, formula: str):
        # We use _ for "protected" attributes
        self._data: pd.DataFrame = data
        self._formula: str = formula
     
        # This will store the fitted model result from statsmodels
        self._result = None 

    @abstractmethod
    def fit(self):
        """
        Fit the statistical model to the data.
        
        This method must be implemented by all subclasses.
        It should store its findings in the `self._result` attribute.
        """
        pass

    @abstractmethod
    def predict(self, new_data: pd.DataFrame):
        """
        Generate predictions using the fitted model on new data.
        """
        pass

    @abstractmethod
    def summary(self):
        """
        Provide a formatted summary of the model's results.
        
        This method will be implemented by subclasses, likely using
        the `summary_formatter.py` module.
        """
        pass

    @abstractmethod
    def plot_diagnostics(self):
        """
        Generate and display diagnostic plots for the model.
        
        This method will be implemented by subclasses, likely using
        the `plot_helper.py` module.
        """
        pass

    def _check_fitted(self):
        """
        Protected helper method to verify if the model has been fitted.
        """
        if self._result is None:
            raise ModelNotFittedError()

    
    @property
    def data(self) -> pd.DataFrame:
        """Get the model's training data."""
        return self._data

    @property
    def formula(self) -> str:
        """Get the model's formula string."""
        return self._formula

    @property
    def result(self):
        """Get the raw fitted model object (from statsmodels)."""
        self._check_fitted() # Ensure model is fitted before returning result
        return self._result

    def __repr__(self) -> str:
        """Standard representation of the model object."""
        return f"{self.__class__.__name__}(formula='{self._formula}')"