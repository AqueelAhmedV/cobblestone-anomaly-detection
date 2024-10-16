# custom exceptions for more granular error  handling
class AnomalyDetectionError(Exception):
    """Base class for exceptions in this module."""
    pass

class InvalidParameterError(AnomalyDetectionError):
    """Exception raised for invalid input parameters."""
    pass

class DataProcessingError(AnomalyDetectionError):
    """Exception raised for errors during data processing."""
    pass

class ModelEvaluationError(AnomalyDetectionError):
    """Exception raised for errors during model evaluation."""
    pass
