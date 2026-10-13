from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Deepfake Image Detection API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173", "*"]
    
    # Database
    DATABASE_URL: str = "sqlite:///./deepfake.db"
    
    # ML Model Config
    MODEL_NAME: str = "resnet50"
    MODEL_VERSION: str = "1.0.0"
    MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "../../../ml/models/best_model.pth")
    IMAGE_SIZE: int = 224
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # File Limits
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
