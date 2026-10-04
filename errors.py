"""Errores controlados del índice Pinecone, la ingesta y la evaluación."""

from __future__ import annotations


class RAGCloudError(Exception):
    """Fallo de índice, ingesta o recuperación convertido en error de aplicación."""


class ClavePineconeError(RAGCloudError):
    """401/key: falta la API key o Pinecone/OpenAI la rechazó."""


class CuotaPineconeError(RAGCloudError):
    """429/cuota: rate limit o cupo del proyecto."""


class RedPineconeError(RAGCloudError):
    """red/timeout: red, timeout o el índice no llegó a ready."""


class DimensionMismatchError(RAGCloudError):
    """El índice o el vector no coinciden con la dimensión pedida."""


class MetricaMismatchError(RAGCloudError):
    """La métrica del índice no es la del embedding (cosine)."""


class EsquemaMetadatosError(RAGCloudError):
    """Metadata fuera del contrato (schema drift, namespace o campo inválido)."""


class SalidaTruncadaError(RAGCloudError):
    """El texto del chunk no entra en la metadata y no se recorta en silencio."""


class ConsultaVaciaError(RAGCloudError):
    """La consulta no tiene texto para embeddear ni buscar."""


class LoteInvalidoError(RAGCloudError):
    """batch_size, k, chunk_size o dimensión fuera de rango."""


class IngestaError(RAGCloudError):
    """Carpeta vacía, splitter sin fragmentos o namespace sin documentos."""
