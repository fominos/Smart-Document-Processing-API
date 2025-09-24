import io
import base64
import re
import logging
from typing import Dict, Any, Tuple, Optional, List
from app.services import yandex_vision
from app.services.stamp_detection import detect_stamps_and_signatures
from app.services.word_processor import process_word_file

logger = logging.getLogger(__name__)


async def process_ttn_loading(file_content: str, file_name: str, file_type: str) -> Dict[str, Any]:
    """
    Единая обработка ТТН за один вызов Vision
    """
    try:
        logger.info(f"Обработка ТТН: {file_name}")

        # 1. Декодируем файл
        file_bytes = base64.b64decode(file_content)
        file_type_lower = file_type.lower()

        # 2. Определяем тип обработки
        if any(ext in file_type_lower for ext in ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp']):
            # Для изображений/PDF - один вызов Vision для всего
            return await _process_vision_ttn(file_bytes, file_type, file_name)

        elif any(ext in file_type_lower for ext in ['docx', 'doc']):
            # Для Word - конвертируем и обрабатываем
            return await _process_word_ttn(file_bytes, file_name)

        else:
            raise ValueError(f"Неподдерживаемый тип файла: {file_type}")

    except Exception as e:
        logger.error(f"Ошибка обработки ТТН: {str(e)}")
        return {
            "error": True,
            "error_text": f"Ошибка обработки: {str(e)}",
            "total_stamps": 0,
            "total_signs": 0,
            "stamps": 0,
            "signs": 0
        }


async def _process_vision_ttn(file_bytes: bytes, file_type: str, file_name: str) -> Dict[str, Any]:
    """
    Обрабатывает изображение/PDF за один вызов Vision
    """
    try:
        # 1. Вызываем Vision для распознавания текста с координатами
        vision_result = await yandex_vision.recognize_image(file_bytes)

        if not vision_result:
            return {
                "error": True,
                "error_text": "Не удалось распознать документ",
                "total_stamps": 0,
                "total_signs": 0,
                "stamps": 0,
                "signs": 0
            }

        # 2. Извлекаем текст и проверяем, что это ТТН
        text = _extract_text_from_vision_result(vision_result)
        is_ttn = is_transport_nakladnaya(text)

        if not is_ttn:
            return {
                "error": True,
                "error_text": "Файл не является транспортной накладной",
                "total_stamps": 0,
                "total_signs": 0,
                "stamps": 0,
                "signs": 0
            }

        # 3. Ищем координаты "Прием груза" в результатах Vision
        section_position = _find_section_in_vision_result(vision_result,
                                                          ["Прием груза", "Приём груза"])

        # 4. Детекция печатей и подписей
        file_content_base64 = base64.b64encode(file_bytes).decode()
        detection_result = await detect_stamps_and_signatures(
            file_content_base64,
            file_name,
            file_type
        )

        # 5. Анализируем результаты
        return await _analyze_ttn_results(detection_result, section_position, vision_result)

    except Exception as e:
        logger.error(f"Ошибка обработки Vision ТТН: {str(e)}")
        return {
            "error": True,
            "error_text": f"Ошибка обработки: {str(e)}",
            "total_stamps": 0,
            "total_signs": 0,
            "stamps": 0,
            "signs": 0
        }


async def _process_word_ttn(file_bytes: bytes, file_name: str) -> Dict[str, Any]:
    """Обработка Word документов"""
    try:
        # Конвертируем Word в текст и проверяем, что это ТТН
        file_content_base64 = base64.b64encode(file_bytes).decode()
        text = await process_word_file(file_content_base64, 'docx')

        is_ttn = is_transport_nakladnaya(text)

        if not is_ttn:
            return {
                "error": True,
                "error_text": "Файл не является транспортной накладной",
                "total_stamps": 0,
                "total_signs": 0,
                "stamps": 0,
                "signs": 0
            }

        # Для Word используем эвристический подход (без координат)
        detection_result = await detect_stamps_and_signatures(
            file_content_base64,
            file_name,
            'docx'
        )

        # Анализируем с эвристикой
        return await _analyze_ttn_results(detection_result, None, None)

    except Exception as e:
        logger.error(f"Ошибка обработки Word ТТН: {str(e)}")
        return {
            "error": True,
            "error_text": f"Ошибка обработки Word: {str(e)}",
            "total_stamps": 0,
            "total_signs": 0,
            "stamps": 0,
            "signs": 0
        }


def _extract_text_from_vision_result(vision_result: Dict) -> str:
    """Извлекает текст из результата Vision"""
    try:
        text_parts = []

        for page_result in vision_result.get("results", []):
            for detection_result in page_result.get("results", []):
                text_detection = detection_result.get("textDetection", {})

                for page in text_detection.get("pages", []):
                    for block in page.get("blocks", []):
                        block_text = []
                        for line in block.get("lines", []):
                            line_text = " ".join(word.get("text", "") for word in line.get("words", []))
                            block_text.append(line_text)
                        text_parts.append("\n".join(block_text))

        return "\n\n".join(text_parts)

    except Exception as e:
        logger.error(f"Ошибка извлечения текста: {str(e)}")
        return ""


def _find_section_in_vision_result(vision_result: Dict, search_phrases: List[str]) -> Optional[Dict]:
    """Ищет фразу в результатах Vision и возвращает координаты"""
    try:
        for page_result in vision_result.get("results", []):
            for detection_result in page_result.get("results", []):
                text_detection = detection_result.get("textDetection", {})

                for page in text_detection.get("pages", []):
                    page_width = page.get("width", 0)
                    page_height = page.get("height", 0)

                    for block in page.get("blocks", []):
                        for line in block.get("lines", []):
                            line_text = " ".join(word.get("text", "") for word in line.get("words", []))

                            for phrase in search_phrases:
                                if phrase.lower() in line_text.lower():
                                    bbox = _calculate_line_bbox(line, page_width, page_height)
                                    if bbox:
                                        return {
                                            "text": line_text,
                                            "phrase": phrase,
                                            "bbox": bbox,
                                            "page_width": page_width,
                                            "page_height": page_height,
                                            "found": True
                                        }
        return None

    except Exception as e:
        logger.error(f"Ошибка поиска секции: {str(e)}")
        return None


def _calculate_line_bbox(line: Dict, page_width: int, page_height: int) -> Optional[List[float]]:
    """Вычисляет bounding box для строки текста с преобразованием типов"""
    try:
        words = line.get("words", [])
        if not words:
            return None

        x_coords = []
        y_coords = []

        for word in words:
            bounding_box = word.get("boundingBox", {})
            vertices = bounding_box.get("vertices", [])
            if len(vertices) >= 4:
                # Преобразуем координаты в float
                x1 = float(vertices[0].get("x", 0))
                y1 = float(vertices[0].get("y", 0))
                x2 = float(vertices[2].get("x", 0))
                y2 = float(vertices[2].get("y", 0))

                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])

        if not x_coords or not y_coords:
            return None

        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    except Exception as e:
        logger.error(f"Ошибка вычисления bbox: {str(e)}")
        return None


async def _analyze_ttn_results(detection_result: Dict,
                               section_position: Optional[Dict],
                               vision_result: Dict) -> Dict[str, Any]:
    """Анализирует результаты детекции с преобразованием типов"""
    try:
        # Отладочное логирование
        logger.info(f"Section position: {section_position}")
        if section_position and section_position.get("found"):
            logger.info(f"Section bbox: {section_position['bbox']}, type: {type(section_position['bbox'][0])}")

        if detection_result and detection_result.get("results", {}).get("details"):
            first_obj = detection_result["results"]["details"][0]
            logger.info(f"First object bbox: {first_obj.get('bbox')}, type: {type(first_obj.get('bbox')[0])}")

        if not detection_result.get("success"):
            return {
                "error": True,
                "error_text": "Ошибка детекции печатей и подписей",
                "total_stamps": 0,
                "total_signs": 0,
                "stamps": 0,
                "signs": 0
            }

        results = detection_result.get("results", {})
        details = results.get("details", [])

        total_stamps = results.get("stamp", 0)
        total_signs = results.get("sign", 0)

        stamps_in_section = 0
        signs_in_section = 0

        if section_position and section_position.get("found"):
            # Используем точные координаты "Прием груза"
            section_bbox = section_position["bbox"]
            # Преобразуем координаты секции в float
            section_bottom = float(section_bbox[3]) if len(section_bbox) > 3 else 0
            logger.info(f"=== DEBUG FILTERING ===")
            logger.info(f"Section bottom: {section_bottom}")
            for i, obj in enumerate(details):
                bbox = obj.get("bbox", [])
                if len(bbox) == 4:
                    obj_center_y = (safe_float_conversion(bbox[1]) + safe_float_conversion(bbox[3])) / 2
                    logger.info(f"Obj {i}: {obj['class']} at Y={obj_center_y}, below={obj_center_y > section_bottom}")
            for obj in details:
                bbox = obj.get("bbox", [])
                if len(bbox) == 4:
                    try:
                        # Преобразуем координаты детекции в float
                        obj_top = safe_float_conversion(bbox[1])
                        obj_bottom = safe_float_conversion(bbox[3])
                        obj_center_y = (obj_top + obj_bottom) / 2

                        # Объект должен быть ниже нижней границы текста "Прием груза"
                        if obj_center_y > section_bottom:
                            if obj["class"] == "stamp":
                                stamps_in_section += 1
                            elif obj["class"] == "sign":
                                signs_in_section += 1
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Ошибка преобразования координат объекта: {e}")
                        continue
        else:
            # Эвристика: считаем что подписи в нижних 30% документа
            page_height = _get_page_height_from_vision(vision_result) if vision_result else 1000
            section_y_min = float(page_height) * 0.7

            for obj in details:
                bbox = obj.get("bbox", [])
                if len(bbox) == 4:
                    try:
                        # Преобразуем координаты детекции в float
                        obj_top = safe_float_conversion(bbox[1])
                        obj_bottom = safe_float_conversion(bbox[3])
                        obj_center_y = (obj_top + obj_bottom) / 2

                        if obj_center_y > section_y_min:
                            if obj["class"] == "stamp":
                                stamps_in_section += 1
                            elif obj["class"] == "sign":
                                signs_in_section += 1
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Ошибка преобразования координат объекта: {e}")
                        continue

        # Проверка условий валидности
        error = False
        error_text = ""

        if stamps_in_section == 0 and signs_in_section < 2:
            error = True
            error_text = "При отсутствии печати должно быть не менее 2 подписей в зоне приемки"
        elif signs_in_section < 2:
            error = True
            error_text = "В зоне приемки должно быть не менее 2 подписей"

        return {
            "error": error,
            "error_text": error_text,
            "total_stamps": total_stamps,
            "total_signs": total_signs,
            "stamps": stamps_in_section,
            "signs": signs_in_section,
            "section_found": section_position is not None and section_position.get("found", False)
        }

    except Exception as e:
        logger.error(f"Ошибка анализа результатов: {str(e)}")
        return {
            "error": True,
            "error_text": f"Ошибка анализа: {str(e)}",
            "total_stamps": 0,
            "total_signs": 0,
            "stamps": 0,
            "signs": 0
        }

def _get_page_height_from_vision(vision_result: Dict) -> int:
    """Получает высоту страницы из результата Vision с преобразованием типов"""
    try:
        for page_result in vision_result.get("results", []):
            for detection_result in page_result.get("results", []):
                text_detection = detection_result.get("textDetection", {})
                for page in text_detection.get("pages", []):
                    height = page.get("height", "1000")
                    return safe_int_conversion(height, 1000)# Преобразуем в int
        return 1000
    except (ValueError, TypeError) as e:
        logger.error(f"Ошибка преобразования высоты страницы: {e}")
        return 1000


def is_transport_nakladnaya(text: str) -> bool:
    """Проверяет, является ли документ транспортной накладной"""
    try:
        text_lower = text.lower()
        # Ключевые слова, характерные для транспортных накладных
        keywords = [
            "транспортная накладная",
            "прием груза",
            "приём груза"
        ]

        # Если есть хотя бы 2 ключевых слова - считаем ТТН
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return matches >= 1

    except Exception:
        return False


async def extract_text_from_content(content: str, file_type: str, file_name: str) -> str:
    """
    Извлекает текст из base64 контента в зависимости от типа файла
    """
    try:
        file_type_lower = file_type.lower()

        # Word файлы
        if any(word_type in file_type_lower for word_type in ['docx', 'doc']):
            logger.info(f"Обработка Word файла: {file_type}")
            return await process_word_file(content, file_type)

        # PDF и изображения (через Yandex Vision)
        elif any(vision_type in file_type_lower for vision_type in [
            'pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp'
        ]):
            logger.info(f"Обработка через Yandex Vision: {file_type}")
            return await yandex_vision.process_file_content(content, file_type)

        else:
            raise ValueError(f"Неподдерживаемый тип файла: {file_type}")

    except Exception as e:
        logger.error(f"Ошибка извлечения текста: {str(e)}")
        return ""


# Функция для тестирования (опционально)
async def test_ttn_processing():
    """
    Тестовая функция для проверки обработки ТТН
    """
    # Здесь можно добавить тестовые данные
    pass
def safe_float_conversion(value, default=0.0):
    """Безопасное преобразование в float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value, default=0):
    """Безопасное преобразование в int"""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default