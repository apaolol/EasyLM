class EasyLMError(Exception):
    """Base exception for all EasyLM-specific errors."""
    pass

class ModelNotFittedError(EasyLMError):
    """Exception raised when trying to use a model that has not been fitted."""
    def __init__(self, message="This model has not been fitted yet. Call .fit() before using this method."):
        self.message = message
        super().__init__(self.message)

class InvalidModelError(EasyLMError):
    """Exception raised when an invalid model type is used."""
    pass