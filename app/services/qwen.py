import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any
from time import time
from app.config import settings
from app.services.iam_token import token_manager

logger = logging.getLogger(__name__)

class QwenClient:
    def __init__(self):
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))

    async def close(self):
        await self.session.close()

    async def analyze_document(
        self,
        content: str,       # base64 без префикса
        prompt: str,
        system_prompt: str,
        mime_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        start_time = time()
        try:
            logger.info("Начинаем обработку запроса к Qwen")
            iam_token = await token_manager.get_token()

            payload = {
                "modelUri": f"gpt://{settings.YANDEX_VISION_FOLDER_ID}/qwen2.5-vl-32b-instruct/latest",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.1,
                    "maxTokens": 2000
                },
                "messages": [
                    {
                        "role": "system",
                        "text": system_prompt + "\n\nФормат ответа: строго JSON"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image": {
                                    "data": content,
                                    "mimeType": mime_type
                                }
                            },
                            {
                                "type": "input_text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            async with self.session.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {iam_token}",
                    "Content-Type": "application/json",
                    # опционально, но иногда помогает при диагностике
                    "x-folder-id": settings.YANDEX_VISION_FOLDER_ID,
                },
                json=payload,
            ) as response:
                resp_text = await response.text()
                if response.status != 200:
                    # Логируем тело ошибки и request-id для поддержки
                    req_id = response.headers.get("x-request-id", "")
                    logger.error(f"YA FM error {response.status}, x-request-id={req_id}, body={resp_text}")
                    return {
                        "extracted_data": {},
                        "violations": [f"Ошибка FM API {response.status}: {resp_text}"],
                        "success": False
                    }

                result = json.loads(resp_text)
                # Парсим текст из alternatives
                response_text = (
                    result.get("result", {})
                          .get("alternatives", [{}])[0]
                          .get("message", {})
                          .get("text", "")
                )

                try:
                    result_data = json.loads(response_text)
                    return {
                        "extracted_data": result_data.get("data", {}),
                        "violations": result_data.get("violations", []),
                        "success": True
                    }
                except json.JSONDecodeError:
                    return {
                        "extracted_data": {},
                        "violations": ["Ошибка формата ответа от модели"],
                        "success": False
                    }
        except Exception as e:
            logger.exception("Ошибка при работе с Qwen")
            return {
                "extracted_data": {},
                "violations": [f"Системная ошибка: {str(e)}"],
                "success": False
            }
        finally:
            logger.info(f"Общая длительность обработки: {time() - start_time:.2f} секунд")

qwen_client = QwenClient()

async def close_qwen_client():
    await qwen_client.close()

