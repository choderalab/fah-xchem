from ..schema import DataPath


class AnalysisError(RuntimeError):
    pass


class ConfigurationError(AnalysisError):
    pass


class DataValidationError(AnalysisError):
    def __init__(self, message: str, path: DataPath):
        super().__init__(message)
        self.path = path


class InsufficientDataError(AnalysisError):
    pass


class InvalidResultError(AnalysisError):
    pass
