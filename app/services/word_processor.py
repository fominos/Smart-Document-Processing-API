import io
import base64
import logging
from typing import Optional
import docx2txt
from docx import Document


logger = logging.getLogger(__name__)


async def extract_text_from_docx(content: str) -> str:
    """
    Извлекает текст из DOCX файлов

    Args:
        content: base64 encoded content

    Returns:
        Извлеченный текст
    """
    try:
        # Декодируем base64
        file_bytes = base64.b64decode(content)

        # Обработка .docx
        with io.BytesIO(file_bytes) as file_stream:
            text = docx2txt.process(file_stream)
            return text.strip()

    except Exception as e:
        logger.error(f"Ошибка при обработке DOCX файла: {e}")
        raise ValueError(f"Не удалось обработать DOCX файл: {str(e)}")


async def extract_text_from_doc(content: str) -> str:
    """
    Извлекает текст из DOC файлов (старый формат)

    Args:
        content: base64 encoded content

    Returns:
        Извлеченный текст
    """
    try:
        # Декодируем base64
        file_bytes = base64.b64decode(content)

        # Для .doc файлов используем ту же библиотеку, но с осторожностью
        with io.BytesIO(file_bytes) as file_stream:
            try:
                # Пытаемся обработать как DOCX (иногда срабатывает)
                text = docx2txt.process(file_stream)
                return text.strip()
            except Exception:
                # Если не получилось, возвращаем сообщение об ошибке
                logger.warning("DOC файл не может быть обработан напрямую")
                return "Не удалось извлечь текст из DOC файла. Рекомендуется конвертировать в DOCX формат."

    except Exception as e:
        logger.error(f"Ошибка при обработке DOC файла: {e}")
        raise ValueError(f"Не удалось обработать DOC файл: {str(e)}")

async def process_word_file(content: str, file_type: str) -> str:
    """
    Универсальный обработчик Word файлов

    Args:
        content: base64 encoded content
        file_type: MIME type файла

    Returns:
        Извлеченный текст
    """
    file_type_lower = file_type.lower()

    if any(docx_type in file_type_lower for docx_type in [
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.docx',
        'docx'
    ]):
        logger.info("Обработка DOCX файла")
        return await extract_text_from_docx(content)

    elif any(doc_type in file_type_lower for doc_type in [
        'application/msword',
        '.doc',
        'doc'
    ]):
        logger.info("Обработка DOC файла")
        return await extract_text_from_doc(content)

    else:
        raise ValueError(f"Unsupported Word file type: {file_type}")