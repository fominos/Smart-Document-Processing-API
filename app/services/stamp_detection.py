import aiohttp
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def detect_stamps_and_signatures(file_content: str, filename: str, file_type: str) -> Dict[str, Any]:
    """
    Отправляет файл в base64 в API детекции печатей и подписей
    с правильным именем файла с расширением
    """
    try:
        # Формируем полное имя файла с расширением
        full_filename = get_filename_with_extension(filename, file_type)

        # Подготавливаем данные для запроса
        payload = {
            "filename": full_filename,  # Полное имя с расширением
            "data": file_content  # Base64 строка
        }

        logger.info(f"Отправка файла в API детекции: {full_filename}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://stampdetection-fominos.amvera.io/detect-binary",
                    json=payload,
                    timeout=60
            ) as response:
                response.raise_for_status()
                result = await response.json()

                logger.info(f"Stamp detection API response: {result}")
                return result

    except aiohttp.ClientError as e:
        logger.error(f"Ошибка при запросе к API детекции: {str(e)}")
        return {
            "success": False,
            "error": f"Ошибка API детекции: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Неожиданная ошибка при детекции: {str(e)}")
        return {
            "success": False,
            "error": f"Неожиданная ошибка: {str(e)}"
        }


def get_filename_with_extension(filename: str, file_type: str) -> str:
    """
    Формирует полное имя файла с расширением на основе file_type
    """
    # Маппинг MIME types к расширениям
    mime_to_extension = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/msword': '.doc',
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg',
        'image/png': '.png',
        'image/tiff': '.tiff',
        'image/bmp': '.bmp',
        'image/webp': '.webp'
    }

    # Пытаемся получить расширение из MIME type
    extension = mime_to_extension.get(file_type.lower())

    # Если не нашли по MIME, пробуем извлечь из file_type
    if not extension:
        if '.' in file_type:
            extension = file_type.split('.')[-1]
            if extension:  # Убеждаемся, что извлекли что-то
                extension = f".{extension}"
        else:
            extension = file_type  # Используем как есть

    # Убеждаемся, что extension начинается с точки
    if extension and not extension.startswith('.'):
        extension = f".{extension}"

    # Убираем существующее расширение из filename (если есть) и добавляем правильное
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    return f"{base_name}{extension}"