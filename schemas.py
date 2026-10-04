"""Contrato Pydantic de entorno, metadatos del chunk y golden set."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigEntorno(BaseModel):
    """Variables del `.env`: no se imprimen ni se commitean con valor real."""

    model_config = ConfigDict(extra="ignore")

    PINECONE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    INDEX_NAME: str = "techcorp-rag-hibrido"

    @field_validator("INDEX_NAME")
    @classmethod
    def index_name_no_vacio(cls, valor: str) -> str:
        limpio = (valor or "").strip()
        if not limpio:
            raise ValueError("INDEX_NAME vacío.")
        return limpio


class MetadatosChunk(BaseModel):
    """Lo que viaja en la metadata: texto original, archivo y categoría."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    categoria: str = Field(min_length=1)
    chunk_id: int = Field(ge=0)


class ItemGolden(BaseModel):
    """Par del benchmark: pregunta y documento que tiene que aparecer en el top-k."""

    model_config = ConfigDict(extra="forbid")

    pregunta: str = Field(min_length=1)
    documento_id_esperado: str = Field(min_length=1)

    @field_validator("pregunta", "documento_id_esperado")
    @classmethod
    def no_blanco(cls, valor: str) -> str:
        limpio = valor.strip()
        if not limpio:
            raise ValueError("El campo no puede quedar vacío.")
        return limpio


class DetallePregunta(BaseModel):
    pregunta: str
    documento_id_esperado: str
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    recuperados: List[str]


class ResultadoEvaluacion(BaseModel):
    """Promedio macro de Precision@k y Recall@k sobre el golden set."""

    k: int = Field(ge=1)
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    detalle: List[DetallePregunta]
