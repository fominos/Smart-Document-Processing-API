import requests
import aiohttp
import asyncio
import base64
from io import BytesIO
from pdf2image import convert_from_bytes
from app.config import settings
import logging
from .iam_token import token_manager

logger = logging.getLogger(__name__)


async def recognize_image(image_bytes: bytes, max_retries: int = 3) -> str:
    """Распознает текст из бинарных данных изображения"""
    last_exception = None
    for attempt in range(max_retries):
        try:
            # Получаем свежий IAM-токен
            iam_token = await token_manager.get_token()
            # Кодируем в base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Формируем запрос
            payload = {
                "folderId": settings.YANDEX_VISION_FOLDER_ID,
                "analyze_specs": [{
                    "content": image_base64,
                    "features": [{
                        "type": "TEXT_DETECTION",
                        "text_detection_config": {
                            "language_codes": ["ru", "en"],
                            "model": "page"
                        }
                    }]
                }]
            }

            # Отправка асинхронного запроса
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        "https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze",
                        headers={
                            "Authorization": f"Bearer {iam_token}",
                            "Content-Type": "application/json"
                        },
                        json=payload,
                        timeout=30
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    logger.debug(f"Raw response from Yandex Vision API: {data}")
                    return data


        except aiohttp.ClientError as e:
            last_exception = e
            if e.status == 429:  # Too Many Requests
                wait_time = 2 ** attempt  # Экспоненциальный backoff
                logger.warning(
                    f"Vision API 429 (Too Many Requests), attempt {attempt + 1}/{max_retries}, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"HTTP error {e.status} from Vision API: {e.message}")
                break  # Не повторяем для других HTTP ошибок
        except aiohttp.ClientError as e:
            last_exception = e
            logger.warning(f"Network error on attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 1 * (attempt + 1)
                await asyncio.sleep(wait_time)
        except Exception as e:
            last_exception = e
            logger.error(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {str(e)}")
            break  # Не повторяем для непредвиденных ошибок
    if isinstance(last_exception, aiohttp.ClientResponseError):
        if last_exception.status == 429:
            error_msg = f"Превышен лимит запросов к Vision API после {max_retries} попыток"
        else:
            error_msg = f"Ошибка Vision API {last_exception.status}: {last_exception.message}"
    elif isinstance(last_exception, aiohttp.ClientError):
        error_msg = f"Ошибка сети при запросе к Vision API после {max_retries} попыток: {str(last_exception)}"
    else:
        error_msg = f"Неизвестная ошибка при распознавании текста после {max_retries} попыток: {str(last_exception)}"

    logger.error(error_msg)
    raise ValueError(error_msg)


async def process_pdf(pdf_bytes: bytes) -> str:
    """Обрабатывает PDF файл"""
    try:
        # Конвертируем PDF в изображения
        images = convert_from_bytes(pdf_bytes)
        full_text = []
        structured_text = []
        # Обрабатываем каждую страницу
        for i, image in enumerate(images):
            logger.info(f"Обработка страницы {i + 1}")
            width, height = image.size
            if width * height > 50000000:  # Проверяем лимит API
                new_width = int(width * 0.7)
                new_height = int(height * 0.7)
                image = image.resize((new_width, new_height))
                logger.info(f"Уменьшено изображение со {width}x{height} до {new_width}x{new_height}")
            # Конвертируем изображение в bytes
            with BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                image_bytes = buffer.getvalue()

            # Распознаем текст
            result = await recognize_image(image_bytes)
            if result:
                # Извлекаем текст из результата
                page_text = []  # Текст для одной страницы
                for page_result in result.get("results", []):
                    for detection_result in page_result.get("results", []):
                        text_detection = detection_result.get("textDetection", {})
                        for page in text_detection.get("pages", []):
                            for block in page.get("blocks", []):
                                block_text = []  # Текст для одного блока
                                for line in block.get("lines", []):
                                    line_text = " ".join(word.get("text", "") for word in line.get("words", []))
                                    block_text.append(line_text)  # Добавляем строку в блок
                                page_text.append("\n".join(block_text))  # Добавляем блок в страницу
                structured_text.append("\n\n".join(page_text))  # Добавляем страницу в общий текст

        # Объединяем текст всех страниц с разделением
        full_text_combined = "\n\n=== Новая страница ===\n\n".join(structured_text)
        return full_text_combined

    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        raise ValueError(f"Ошибка обработки PDF: {str(e)}")


async def process_file_content(content: str, file_type: str) -> str:
    """Основная функция обработки"""
    try:
        # Декодируем base64
        file_bytes = base64.b64decode(content)

        if file_type.lower() == "pdf":
            return await process_pdf(file_bytes)
        else:
            # Для изображений (jpg, png и т.д.)
            structured_text = []
            result = await recognize_image(file_bytes)
            if result:
                # Извлекаем текст из результата
                page_text = []  # Текст для одной страницы
                for page_result in result.get("results", []):
                    for detection_result in page_result.get("results", []):
                        text_detection = detection_result.get("textDetection", {})
                        for page in text_detection.get("pages", []):
                            for block in page.get("blocks", []):
                                block_text = []  # Текст для одного блока
                                for line in block.get("lines", []):
                                    line_text = " ".join(word.get("text", "") for word in line.get("words", []))
                                    block_text.append(line_text)  # Добавляем строку в блок
                                page_text.append("\n".join(block_text))  # Добавляем блок в страницу
                structured_text.append("\n\n".join(page_text))  # Добавляем страницу в общий текст
            full_text_combined = "\n\n".join(structured_text)
            return full_text_combined

    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        raise ValueError(f"Ошибка обработки файла: {str(e)}")

