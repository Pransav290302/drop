

class DropSmartException(Exception):
    
    pass


class FileValidationError(DropSmartException):
    
    pass


class SchemaValidationError(DropSmartException):
    
    pass


class ProcessingError(DropSmartException):
    
    pass


class ModelNotFoundError(DropSmartException):
    
    pass


class ModelPredictionError(DropSmartException):
    
    pass

