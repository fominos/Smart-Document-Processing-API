import logging
from datetime import datetime, timedelta
import aiohttp
import jwt
from pathlib import Path
import time
from app.config import settings
import json
import asyncio

logger = logging.getLogger(__name__)


class IAMTokenManager:
    def __init__(self):
        self.token = None
        self.expires_at = None
        self._key_data = None
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Явная инициализация при старте приложения"""
        try:
            await self._refresh_jwt_token()
            logger.info("Initial IAM token successfully obtained")
        except Exception as e:
            logger.error(f"Failed to get initial IAM token: {str(e)}")
            raise
    def _load_key_data(self) -> dict:
        """Загружает и валидирует ключ из файла"""
        if self._key_data is None:
            try:
                key_path = Path(settings.YANDEX_SERVICE_ACCOUNT_KEY_PATH)
                if not key_path.exists():
                    raise FileNotFoundError(f"Key file not found: {key_path}")

                with open(key_path, "r") as f:
                    self._key_data = json.load(f)

                # Проверка обязательных полей
                required_fields = ["id","service_account_id", "private_key"]
                for field in required_fields:
                    if field not in self._key_data:
                        raise ValueError(f"Missing required field in key file: {field}")

                # Нормализация ключа
                self._key_data["private_key"] = "\n".join(
                    line.strip() for line in self._key_data["private_key"].split("\n")
                )

            except Exception as e:
                logger.error(f"Key file loading error: {str(e)}")
                raise

        return self._key_data

    async def get_token(self) -> str:
        """Получает токен, при необходимости обновляет"""
        async with self._lock:  # Защита от конкурентного доступа
            if self.token is None or datetime.now() >= self.expires_at:
                await self._refresh_jwt_token()
            return self.token

    async def _refresh_jwt_token(self):
        """Генерирует новый IAM-токен"""
        try:
            key_data = self._load_key_data()

            # Формируем JWT
            now = int(time.time())
            jwt_payload = {
                "aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens",
                "iss": key_data["service_account_id"],
                "iat": now,
                "exp": now + 3600  # 1 час
            }
            jwt_headers = {
                "kid": key_data["id"]  # Добавляем обязательное поле kid
            }
            jwt_token = jwt.encode(
                jwt_payload,
                key_data["private_key"],
                algorithm="PS256",
                headers=jwt_headers  # Передаем заголовки
            )

            # Получаем IAM-токен
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        "https://iam.api.cloud.yandex.net/iam/v1/tokens",
                        json={"jwt": jwt_token},
                        timeout=10
                ) as resp:
                    if resp.status != 200:
                        error_body = await resp.text()
                        logger.error(f"IAM token request failed: {resp.status} {error_body}")
                        raise ValueError(f"IAM API error: {resp.status}")

                    data = await resp.json()
                    self.token = data["iamToken"]
                    self.expires_at = datetime.now() + timedelta(hours=11)
                    logger.info(f"IAM token refreshed, expires at {self.expires_at}")
        except jwt.PyJWTError as e:
            logger.error(f"JWT generation error: {str(e)}")
            raise ValueError(f"JWT generation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in _refresh_jwt_token: {str(e)}")
            raise ValueError(f"Failed to refresh IAM token: {str(e)}")

# Глобальный экземпляр менеджера токенов
token_manager = IAMTokenManager()


