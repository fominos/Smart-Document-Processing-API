from pydantic_settings import BaseSettings
from typing import Optional
from datetime import datetime, timedelta

class Settings(BaseSettings):
    YANDEX_SERVICE_ACCOUNT_KEY_PATH: str
    YANDEX_VISION_FOLDER_ID: str
    YANDEX_GPT_FOLDER_ID: str
    iam_token: Optional[str] = None
    iam_token_expires: Optional[datetime] = None
    QWEN_API_URL: str = "https://api.qwen.com/v1/chat"
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()