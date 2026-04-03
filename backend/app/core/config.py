"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "FeelsLike AI"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/feelslike"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/feelslike"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Weather API
    open_meteo_base_url: str = "https://api.open-meteo.com/v1"

    # ML
    model_dir: str = "models"
    base_model_path: str = "models/base_comfort_model.onnx"
    retrain_feedback_threshold: int = 5  # retrain after N new feedbacks

    # Notifications
    max_daily_notifications: int = 2

    model_config = {"env_file": ".env", "env_prefix": "FEELSLIKE_"}


settings = Settings()
