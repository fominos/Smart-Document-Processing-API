from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from typing import Dict, Any, List

class ModelType(str, Enum):
    LITE = "yandexgpt-lite"
    PRO = "yandexgpt"
    QWEN = "qwen"
class AnalysisResult(BaseModel):
    text: str
    analysis: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_estimate: Optional[float] = None



class ContractAnalysisResult(BaseModel):
    extracted_data: Optional[Dict] = Field(default_factory=dict)
    violations: Optional[List[str]] = Field(default_factory=list)
    error: Optional[bool] = None
    error_text: Optional[str] = None
    total_stamps: Optional[int] = None
    total_signs: Optional[int] = None
    stamps: Optional[int] = None
    signs: Optional[int] = None

