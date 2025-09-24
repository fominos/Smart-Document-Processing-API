import logging
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """
    Анализирует расположение печатей и подписей относительно текстовых меток
    """

    @staticmethod
    async def analyze_ttn_signatures(text: str, detection_results: Dict[str, Any], image_height: int = 1200) -> Dict[
        str, Any]:
        """
        Анализирует расположение подписей и печатей относительно пункта 8
        Возвращает статистику в формате для основного API
        """
        try:
            # Ищем позицию пункта 8 в тексте
            section_8_position = await LayoutAnalyzer.find_section_position(text, "Прием груза", "Приём груза",
                                                                            "пункт 8", "8.")

            if not section_8_position:
                logger.warning("Не найден прием груза в документе, используем консервативную зону")
                section_8_position = {
                    "y_min": 0.7,  # последние 30% документа
                    "y_max": 1.0,
                    "section_line": 0.7,
                    "section_name": "conservative_zone"
                }

            # Анализируем расположение объектов детекции
            analysis_results = await LayoutAnalyzer.analyze_detection_objects(
                detection_results, section_8_position, image_height
            )

            # Преобразуем в формат для основного API
            return await LayoutAnalyzer.format_for_main_api(analysis_results)

        except Exception as e:
            logger.error(f"Ошибка анализа расположения: {str(e)}", exc_info=True)
            return {
                "error": True,
                "error_text": f"Ошибка анализа: {str(e)}",
                "total_stamps": 0,
                "total_signs": 0,
                "stamps": 0,
                "signs": 0
            }

    @staticmethod
    async def find_section_position(text: str, *section_patterns: str) -> Optional[Dict[str, float]]:
        """
        Находит точную позицию 'Приема груза' и возвращает зону СТРОГО ПОСЛЕ него
        """
        try:
            text_lower = text.lower()
            lines = text_lower.split('\n')

            for i, line in enumerate(lines):
                for pattern in section_patterns:
                    pattern_lower = pattern.lower()
                    if pattern_lower in line:
                        total_lines = len(lines)
                        if total_lines == 0:
                            return None

                        # Позиция строки с "Прием груза"
                        section_line_position = i / total_lines

                        # Зона подписей начинается СРАЗУ ПОСЛЕ этой строки
                        section_start = min((i + 1) / total_lines, 0.9)

                        # Узкая зона - только следующие 15% документа
                        section_end = min(section_start + 0.15, 1.0)

                        logger.info(f"Найден '{pattern}' на строке {i}/{total_lines}")

                        return {
                            "y_min": section_start,
                            "y_max": section_end,
                            "section_line": section_line_position,
                            "line_number": i,
                            "total_lines": total_lines,
                            "section_name": pattern
                        }

            return None
        except Exception as e:
            logger.error(f"Ошибка в find_section_position: {e}")
            return None

    @staticmethod
    async def analyze_detection_objects(detection_results: Dict[str, Any],
                                        section_position: Dict[str, float],
                                        image_height: int) -> Dict[str, Any]:
        """
        Анализирует объекты детекции относительно позиции секции
        """
        try:
            # Проверяем успешность детекции
            if not detection_results or not detection_results.get("success", False):
                return {
                    "valid": False,
                    "errors": ["Детекция не выполнена или неуспешна"],
                    "stamp_count": 0,
                    "signature_count": 0,
                    "stamps_in_section": 0,
                    "signs_in_section": 0,
                    "detection_success": False
                }

            results = detection_results.get("results", {})
            details = results.get("details", [])

            # Общее количество
            total_stamps = results.get("stamp", 0)
            total_signs = results.get("sign", 0)

            # Анализируем каждый объект
            stamps_in_section = 0
            signs_in_section = 0

            for obj in details:
                position_analysis = await LayoutAnalyzer.analyze_object_position(
                    obj, section_position, image_height
                )

                if position_analysis.get("valid", False):
                    if obj["class"] == "stamp":
                        stamps_in_section += 1
                    elif obj["class"] == "sign":
                        signs_in_section += 1

            # Проверяем условия валидности ТТН
            errors = []
            valid = True

            # Условие 1: Если нет печати, должно быть не менее 2 подписей
            if stamps_in_section == 0 and signs_in_section < 2:
                errors.append("При отсутствии печати должно быть не менее 2 подписей в зоне приемки")
                valid = False

            # Условие 2: Минимальное количество подписей
            elif signs_in_section < 2:
                errors.append("В зоне приемки должно быть не менее 2 подписей")
                valid = False

            logger.info(f"Анализ завершен: печати={stamps_in_section}, подписи={signs_in_section}, valid={valid}")

            return {
                "valid": valid,
                "errors": errors,
                "stamp_count": total_stamps,
                "signature_count": total_signs,
                "stamps_in_section": stamps_in_section,
                "signs_in_section": signs_in_section,
                "detection_success": True
            }

        except Exception as e:
            logger.error(f"Ошибка в analyze_detection_objects: {e}", exc_info=True)
            return {
                "valid": False,
                "errors": [f"Ошибка анализа объектов: {e}"],
                "stamp_count": 0,
                "signature_count": 0,
                "stamps_in_section": 0,
                "signs_in_section": 0,
                "detection_success": False
            }

    @staticmethod
    async def analyze_object_position(obj: Dict[str, Any],
                                      section_position: Dict[str, float],
                                      image_height: int) -> Dict[str, Any]:
        """
        Анализирует позицию объекта относительно секции
        """
        try:
            bbox = obj.get("bbox", [])
            if len(bbox) != 4:
                return {"valid": False, "error": "Invalid bbox"}

            center_y = (bbox[1] + bbox[3]) / 2
            normalized_y = center_y / image_height

            section_line_pos = section_position.get("section_line", 0)
            section_start = section_position.get("y_min", 0.7)
            section_end = section_position.get("y_max", 1.0)

            # Критическая проверка: объект должен быть СТРОГО ПОСЛЕ строки с пунктом
            is_after_section_line = normalized_y > section_line_pos
            is_in_target_zone = section_start <= normalized_y <= section_end

            is_valid = is_after_section_line and is_in_target_zone

            return {
                "valid": is_valid,
                "center_y": center_y,
                "normalized_y": normalized_y,
                "is_after_section_line": is_after_section_line,
                "is_in_target_zone": is_in_target_zone
            }
        except Exception as e:
            logger.error(f"Ошибка в analyze_object_position: {e}")
            return {"valid": False, "error": str(e)}

    @staticmethod
    async def format_for_main_api(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Преобразует результат анализа в формат для основного API
        """
        try:
            # Если analysis_results уже в правильном формате (с error полем)
            if "error" in analysis_results:
                return analysis_results

            # Если это результат от analyze_detection_objects
            if "valid" in analysis_results:
                return {
                    "error": not analysis_results["valid"],
                    "error_text": "; ".join(analysis_results.get("errors", [])),
                    "total_stamps": analysis_results.get("stamp_count", 0),
                    "total_signs": analysis_results.get("signature_count", 0),
                    "stamps": analysis_results.get("stamps_in_section", 0),
                    "signs": analysis_results.get("signs_in_section", 0)
                }

            # Fallback
            return {
                "error": True,
                "error_text": "Неизвестный формат результата анализа",
                "total_stamps": 0,
                "total_signs": 0,
                "stamps": 0,
                "signs": 0
            }

        except Exception as e:
            logger.error(f"Ошибка в format_for_main_api: {e}")
            return {
                "error": True,
                "error_text": f"Ошибка форматирования: {e}",
                "total_stamps": 0,
                "total_signs": 0,
                "stamps": 0,
                "signs": 0
            }