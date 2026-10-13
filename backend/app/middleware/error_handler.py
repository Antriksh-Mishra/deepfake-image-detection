from fastapi import Request, status
from fastapi.responses import JSONResponse
import logging

from ml.src.preprocessing.exceptions import (
    PreprocessingError,
    InvalidImageError,
    UnsupportedFormatError,
    FileTooLargeError,
    NoFaceDetectedError
)

logger = logging.getLogger(__name__)

async def custom_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler converting Python errors into standardized JSON API error responses."""
    
    if isinstance(exc, NoFaceDetectedError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc),
                "error_code": "NO_FACE_DETECTED",
                "status_code": 400
            }
        )
    elif isinstance(exc, UnsupportedFormatError):
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={
                "detail": str(exc),
                "error_code": "UNSUPPORTED_MEDIA_TYPE",
                "status_code": 415
            }
        )
    elif isinstance(exc, FileTooLargeError):
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "detail": str(exc),
                "error_code": "FILE_TOO_LARGE",
                "status_code": 413
            }
        )
    elif isinstance(exc, InvalidImageError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc),
                "error_code": "INVALID_IMAGE",
                "status_code": 400
            }
        )
    elif isinstance(exc, PreprocessingError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": str(exc),
                "error_code": "PREPROCESSING_ERROR",
                "status_code": 422
            }
        )
    else:
        logger.error(f"Unhandled server exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": f"An internal server error occurred: {str(exc)}",
                "error_code": "INTERNAL_SERVER_ERROR",
                "status_code": 500
            }
        )
