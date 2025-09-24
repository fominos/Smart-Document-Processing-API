import aiohttp
import tiktoken
from app.config import settings
from app.schemas import ModelType
import logging
import json
import re
from .iam_token import token_manager
from typing import List, Dict, Any
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

logger = logging.getLogger(__name__)
encoder = tiktoken.get_encoding("cl100k_base")


class DocumentChunker:
    @staticmethod
    def chunk_document(text: str, max_chunk_size: int = 500) -> List[str]:
        """Разбивает документ на чанки, гарантируя целостность параграфов"""
        # Приоритетные разделители (параграфы, заголовки, списки)
        text = text.replace("=== Новая страница ===", "")
        separators = [
            '\n\n', '\n● ', '\n§ ', '\nСтатья ', '\nПункт ',
            '\n1. ', '\n2. ', '\n3. ', '\n4. ', '\n5. '
        ]

        encoder = tiktoken.get_encoding("cl100k_base")

        # Сначала разбиваем текст на неделимые блоки (параграфы)
        pattern = f"({'|'.join(map(re.escape, separators))})"
        paragraphs = []
        current_para = []

        parts = re.split(pattern, text)
        prev_separator = ""

        for part in parts:
            if not part:
                continue

            if part in separators:
                if current_para:
                    paragraphs.append(''.join(current_para))
                    current_para = []
                prev_separator = part
            else:
                if prev_separator:
                    current_para.append(prev_separator)
                    prev_separator = ""
                current_para.append(part)

        if current_para:
            paragraphs.append(''.join(current_para))

        # Теперь собираем чанки из целых параграфов
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(encoder.encode(para))

            # Если параграф один слишком большой - разбиваем принудительно
            if para_size > max_chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Принудительная разбивка большого параграфа
                words = para.split()
                temp_chunk = []
                temp_size = 0

                for word in words:
                    word_size = len(encoder.encode(word))
                    if temp_size + word_size > max_chunk_size:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                            temp_chunk = []
                            temp_size = 0
                    temp_chunk.append(word)
                    temp_size += word_size

                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                continue

            # Обычный случай - добавляем параграф к текущему чанку
            if current_size + para_size > max_chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_size = 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append(''.join(current_chunk))

        return [chunk for chunk in chunks if chunk.strip()]


class YandexVectorSearch:
    def __init__(self):
        self.embeddings = []
        self.documents = []
        self.document_ids = []
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        self.lock = asyncio.Lock()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Получение эмбеддинга через REST API"""
        logger.debug(f"Токены в тексте: {len(encoder.encode(text))}")
        logger.debug(f"Длина текста: {len(text)} символов")
        iam_token = await token_manager.get_token()

        payload = {
            "modelUri": f"emb://{settings.YANDEX_VISION_FOLDER_ID}/text-search-query",
            "text": text
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {iam_token}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=10
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return np.array(data["embedding"])

    async def remove_document(self, doc_id: str):
        """Удаляет конкретный документ по его идентификатору"""
        async with self.lock:
            if doc_id in self.document_ids:
                index = self.document_ids.index(doc_id)
                self.documents.pop(index)
                self.embeddings.pop(index)
                self.document_ids.pop(index)
                logger.debug(f"Removed document {doc_id}")

    async def index_documents(self, documents: List[str], doc_id: str) -> bool:
        """Индексация документов с привязкой к идентификатору"""
        async with self.lock:
            if doc_id in self.document_ids:
                logger.warning(f"Document {doc_id} already indexed")
                return False

            try:
                embeddings = []
                successful_docs = []
                for i, doc in enumerate(documents):
                    try:
                        embedding = await self._get_embedding(doc)
                        embeddings.append(embedding)
                        successful_docs.append(documents[i])
                        logger.debug(f"Successfully embedded chunk {i} of doc {doc_id}")
                    except Exception as e:
                        logger.error(f"Failed to embed chunk {i} of doc {doc_id}: {str(e)}")
                        continue

                if not embeddings:
                    logger.error(f"No embeddings were generated for doc {doc_id}")
                    return False

                self.documents.extend(successful_docs)
                self.embeddings.extend(embeddings)
                self.document_ids.extend([doc_id] * len(embeddings))

                logger.info(f"Successfully indexed doc {doc_id} with {len(embeddings)} chunks")
                return True

            except Exception as e:
                logger.error(f"Critical indexing failure for {doc_id}: {str(e)}", exc_info=True)
                return False

    async def search(self, query: str, doc_id: str = None, top_k: int = 2) -> List[Dict]:
        """Поиск только среди документов с указанным ID"""
        try:
            query_embedding = await self._get_embedding(query)

            targets = [
                (i, emb) for i, emb in enumerate(self.embeddings)
                if doc_id is None or self.document_ids[i] == doc_id
            ]

            if not targets:
                return []

            scores = [
                np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                for i, emb in targets
            ]

            top_indices = [targets[i][0] for i in np.argsort(scores)[-top_k:][::-1]]
            return [
                {
                    "text": self.documents[i],
                    "score": float(scores[i]),
                    "position": i,
                    "doc_id": self.document_ids[i]
                }
                for i in top_indices
                if scores[i] > 0.3
            ]
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []


async def analyze_contract_with_rag(
        text: str,
        doc_id: str,
        prompt: str,
        system_prompt: str,
        search_aspects: str,
        model: ModelType = ModelType.PRO
) -> dict:
    """
    Анализирует контракт с использованием Yandex Vector Search и GPT
    """
    try:
        logger.info(f"Starting analysis for document {doc_id}, size: {len(text)} chars")
        iam_token = await token_manager.get_token()

        if len(text) > 16000:
            # 1. Подготовка документа
            chunks = DocumentChunker.chunk_document(text)
            logger.info(f"Document {doc_id} split into {len(chunks)} chunks")
            vector_search = YandexVectorSearch()
            index_result = await vector_search.index_documents(chunks, doc_id)

            if not index_result:
                logger.error(f"Failed to index document {doc_id}, using fallback approach")
                context = text[:16000]
            else:
                # 2. Поиск релевантных фрагментов для каждого аспекта
                aspects = [a.strip() for a in re.split(r"[,;\n]", search_aspects) if a.strip()]
                logger.info(f"Searching for {len(aspects)} aspects")
                # Собираем все релевантные фрагменты для каждого аспекта
                aspect_fragments = {}
                for aspect in aspects:
                    results = await vector_search.search(aspect, doc_id)
                    aspect_fragments[aspect] = sorted(results, key=lambda x: x["score"], reverse=True)
                    logger.info(f"Found {len(results)} results for aspect '{aspect}'")

                context_parts = []
                used_positions = set()
                total_context_size = 0
                MAX_CONTEXT_SIZE = 16000

                # Первый проход: по одному лучшему фрагменту на аспект
                for aspect in aspects:
                    if aspect_fragments[aspect]:
                        best_fragment = aspect_fragments[aspect][0]
                        if best_fragment["position"] not in used_positions:
                            fragment_size = len(best_fragment["text"])
                            if total_context_size + fragment_size <= MAX_CONTEXT_SIZE:
                                context_parts.append(best_fragment["text"])
                                used_positions.add(best_fragment["position"])
                                total_context_size += fragment_size
                                logger.info(f"Added primary fragment for aspect '{aspect}'")

                # Второй проход: дополнительные фрагменты, если осталось место
                if total_context_size < MAX_CONTEXT_SIZE:
                    remaining_space = MAX_CONTEXT_SIZE - total_context_size

                    # Собираем все дополнительные фрагменты (кроме уже использованных)
                    additional_fragments = []
                    for aspect in aspects:
                        for fragment in aspect_fragments[aspect][1:]:  # Пропускаем первый (уже использованный)
                            if (fragment["position"] not in used_positions and
                                    fragment["score"] > 0.5):  # Только достаточно релевантные
                                additional_fragments.append(fragment)

                    # Сортируем дополнительные фрагменты по релевантности
                    additional_fragments.sort(key=lambda x: x["score"], reverse=True)

                    # Добавляем пока есть место
                    for fragment in additional_fragments:
                        fragment_size = len(fragment["text"])
                        if total_context_size + fragment_size <= MAX_CONTEXT_SIZE:
                            context_parts.append(fragment["text"])
                            used_positions.add(fragment["position"])
                            total_context_size += fragment_size
                            logger.info(f"Added secondary fragment for aspect")

                await vector_search.remove_document(doc_id)
                context = "\n\n".join(context_parts) if context_parts else text[:16000]
                logger.info(f"Final context: {len(context_parts)} fragments, {len(context)} chars")
        else:
            context = text
            logger.info(f"Using full document: {len(context)} chars")

        # 3. Формирование промпта
        full_prompt = f"""
        {prompt}

       Релевантные фрагменты договора:
        {context}
        """
        # 4. Отправка запроса к Yandex GPT
        payload = {
            "modelUri": f"gpt://{settings.YANDEX_VISION_FOLDER_ID}/{model.value}",
            "messages": [
                {
                    "role": "system",
                    "text": system_prompt + "\n\nФормат ответа: строго JSON"
                },
                {
                    "role": "user",
                    "text": full_prompt
                }
            ],
            "temperature": 0.1,
            "maxTokens": 2000
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                    headers={
                        "Authorization": f"Bearer {iam_token}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=30
            ) as response:
                response.raise_for_status()
                result = await response.json()
                result_text = result["result"]["alternatives"][0]["message"]["text"]

                # Очистка и парсинг ответа
                cleaned_text = clean_json_response(result_text)
                print(cleaned_text)
                try:
                    result_data = json.loads(cleaned_text)

                    # Нормализация ответа
                    return {
                        "extracted_data": result_data.get("data", {}),
                        "violations": result_data.get("violations", [])

                    }
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка парсинга ответа: {str(e)}")
                    logger.error(f"Содержимое cleaned_text, которое не удалось распарсить: {cleaned_text[:50]}")
                    return {
                        "extracted_data": {},
                        "violations": ["[SYSTEM] Ошибка формата ответа"],

                    }

    except Exception as e:
        logger.error(f"Ошибка анализа: {str(e)}", exc_info=True)
        return {
            "extracted_data": {},
            "violations": [f"[SYSTEM] Ошибка обработки: {str(e)}"],
            "context_summary": "Ошибка выполнения"
        }

def clean_json_response(text):
            text = text.strip()
            # Удаляем обратные кавычки если они есть
            if text.startswith('```') and text.endswith('```'):
                text = text[3:-3].strip()
                if text.startswith('json'):
                    text = text[4:].strip()
            text = text.replace('\\', '')
            return text

def force_fix_json(text):
    """Принудительно исправляет самые частые ошибки JSON"""
    # Заменяем все одинарные кавычки на двойные для ключей
    text = re.sub(r"(\w+):\s*'([^']*)'", r'"\1": "\2"', text)  # Ключи: значения
    text = re.sub(r"'(\w+)':", r'"\1":', text)  # Ключи с одинарными кавычками

    # Исправляем несбалансированные кавычки в значениях
    lines = text.split('\n')
    fixed_lines = []

    for line in lines:
        # Если в строке нечетное количество кавычек - исправляем
        if line.count('"') % 2 != 0:
            # Удаляем все кавычки из значений и оставляем только для ключей
            if ':' in line:
                key, value = line.split(':', 1)
                # Очищаем значение от кавычек
                value = value.replace('"', '').strip()
                line = f'{key}: "{value}"'
        fixed_lines.append(line)

    return '\n'.join(fixed_lines)
