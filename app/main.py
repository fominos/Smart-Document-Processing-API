from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.services import yandex_vision, yandex_gpt
from app.services.word_processor import process_word_file
from app.services.ttn_processor import process_ttn_loading
from app.services.qwen import qwen_client
from app.schemas import ContractAnalysisResult, ModelType
from pydantic import BaseModel
from pydantic import Field
import logging
import re
from fastapi.middleware.cors import CORSMiddleware
from app.services.iam_token import token_manager
import uuid
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Минимальный уровень для вывода
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileInfo(BaseModel):
    content: str
    file_type: str
    file_name: str
    file_id: Optional[str] = None

class FileData(BaseModel):
    files: List[FileInfo]  # Теперь принимаем массив файлов
    prompt: str
    system_prompt: str
    search_aspects: str
    model: ModelType

@app.on_event("startup")
async def startup_event():
    """Инициализация при старте приложения"""
    try:
        await token_manager.initialize()  # Явно получаем начальный токен
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise


async def process_file(file_content: str, file_type: str, file_name: str) -> str:
    """
    Универсальная функция обработки файлов
    Определяет тип файла и направляет в соответствующий обработчик
    """
    file_type_lower = file_type.lower()

    # Word файлы
    if any(word_type in file_type_lower for word_type in [
        'docx',
        'doc',
    ]):
        logger.info(f"Обработка Word файла: {file_type}")
        return await process_word_file(file_content, file_type)

    # PDF и изображения (через Yandex Vision)
    elif any(vision_type in file_type_lower for vision_type in [
        'pdf',
        'jpg',
        'jpeg',
        'png',
        'tiff',
        'bmp',
        'webp'
    ]):
        logger.info(f"Обработка через Yandex Vision: {file_type}")
        return await yandex_vision.process_file_content(file_content, file_type)

    else:
        error_msg = f"Неподдерживаемый тип файла: {file_type}"
        logger.warning(error_msg)
        return error_msg

@app.post("/analyze-contract/", response_model=ContractAnalysisResult)
async def analyze_contract(data: FileData):
    try:
        ttn_files = [file for file in data.files
                     if file.file_id and file.file_id.lower() == "ttn_loading"]

        if ttn_files:
            logger.info(f"Найдено {len(ttn_files)} файлов с file_id='ttn_loading'")

            # Обрабатываем все файлы с ttn_loading
            ttn_results = []
            for file in ttn_files:
                ttn_result = await process_ttn_loading(
                    file.content,
                    file.file_name,  # имя без расширения
                    file.file_type  # тип файла/расширение
                )
                ttn_result["file_name"] = file.file_name  # Сохраняем имя файла
                ttn_results.append(ttn_result)

            valid_ttn_results = [result for result in ttn_results if not result.get("error") or
                                 result.get("error_text") != "Файл не является транспортной накладной"]

            if valid_ttn_results:
                # Возвращаем результат первой валидной ТТН
                return ContractAnalysisResult(**valid_ttn_results[0])
            else:
                # Если ни один файл не является ТТН
                return ContractAnalysisResult(
                    error=True,
                    error_text="Ни один из файлов не является транспортной накладной",
                    total_stamps=0,
                    total_signs=0,
                    stamps=0,
                    signs=0
                )
        if data.model == ModelType.QWEN:
            result = await qwen_client.analyze_document(
                content=data.content,
                prompt=data.prompt,
                system_prompt=data.system_prompt
            )
            return ContractAnalysisResult(**result)
        else:
            extracted_texts = []
            seen_texts = set()
            process_errors = []
            for file in data.files:
                # Универсальная обработка файлов
                logger.info(f"Обработка файла: {file.file_name}, тип: {file.file_type}")
                text = await process_file(file.content, file.file_type, file.file_name)
                if text.startswith("Неподдерживаемый тип файла:") or text.startswith("Не удалось извлечь текст из DOC файла"):
                    process_errors.append(text)
                elif text.strip():  # Добавляем только непустой текст
                    normalized_text = re.sub(r'\s+', ' ', text.strip())
                    # Проверяем, не видели ли мы уже этот текст
                    if normalized_text not in seen_texts:
                        seen_texts.add(normalized_text)
                        extracted_texts.append(text)
                    else:
                        logger.info(f"Пропущен дубликат документа: {file.file_name}")
            if process_errors and not extracted_texts:
                error_text = "; ".join(process_errors)
                return ContractAnalysisResult(
                    error=True,
                    error_text=error_text,
                    violations=[],
                    extracted_data={}
                )
            if not extracted_texts:
                    return ContractAnalysisResult(
                        error=True,
                        error_text="Не удалось извлечь текст из файлов",
                        violations=[],
                        extracted_data={}
                    )
                # Объединяем тексты из всех файлов
            combined_text = "\n\n".join(extracted_texts)
            combined_text = re.sub(r'\n{3,}', '\n\n', combined_text)  # Удалить множественные переносы
            # Удаляем разорванные строки
            combined_text = re.sub(r'(\w+)\n(\w+)', r'\1\2', combined_text)
            # Фиксим переносы ИНН
            combined_text = re.sub(r'ИНН\n(\d+)', r'ИНН: \1', combined_text)
            combined_text = "\n".join(line.strip() for line in combined_text.split("\n") if line.strip())
            #extracted_text = extracted_text.replace("Донецк", "Запретная зона")
            #extracted_text = extracted_text.replace("Луганск", "Запретная зона")
            #extracted_text = extracted_text.replace("Мариуполь", "Запретная зона")
            doc_id = str(uuid.uuid4())
            result = await yandex_gpt.analyze_contract_with_rag(combined_text, doc_id, data.prompt, data.system_prompt, data.search_aspects, data.model)
        return ContractAnalysisResult(**result)

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
