class PreprocessingError(Exception):
    """Base exception for preprocessing failures."""
    pass

class InvalidImageError(PreprocessingError):
    """Raised when an uploaded file is not a valid or readable image."""
    pass

class UnsupportedFormatError(PreprocessingError):
    """Raised when an image format is not supported."""
    pass

class FileTooLargeError(PreprocessingError):
    """Raised when an uploaded image exceeds size limits."""
    pass

class NoFaceDetectedError(PreprocessingError):
    """Raised when face detection is enabled but no face is found."""
    pass
