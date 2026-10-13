import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.database import engine, Base
from app.routes import predict
from app.middleware.error_handler import custom_exception_handler
from ml.src.preprocessing.exceptions import PreprocessingError

# Initialize Database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Custom Exception Handler
app.add_exception_handler(Exception, custom_exception_handler)
app.add_exception_handler(PreprocessingError, custom_exception_handler)

# Mount Static Storage directory
storage_abs_dir = os.path.abspath("storage")
os.makedirs(storage_abs_dir, exist_ok=True)
app.mount("/storage", StaticFiles(directory=storage_abs_dir), name="storage")

# Include API Routers
app.include_router(predict.router)

@app.get("/health", tags=["Health"])
@app.get(f"{settings.API_V1_STR}/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model": settings.MODEL_NAME
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
