import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Credit Risk Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database connection URL (defaults to SQLite for local zero-setup dev, PostgreSQL in Docker/production)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./credit_risk.db")
    
    # Model pipeline path
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/credit_risk_pipeline.joblib")
    
    # Host & Port
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
