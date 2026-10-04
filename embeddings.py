"""Embeddings locales: all-MiniLM-L6-v2 (384, cosine) o el sustituto offline del mismo ancho."""

from __future__ import annotations

import hashlib
from typing import List

from langchain_core.embeddings import Embeddings

from errors import DimensionMismatchError

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class DeterministicEmbeddings(Embeddings):
    """Bag-of-words hasheado y normalizado. Misma dimensión en cada corrida, sin bajar modelos."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dimensión inválida: {dim}")
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(texto) for texto in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)

    def _vector(self, texto: str) -> List[float]:
        vec = [0.0] * self.dim
        for crudo in (texto or "").lower().replace("¿", " ").replace("?", " ").split():
            token = "".join(ch for ch in crudo if ch.isalnum() or ch in "-_")
            if len(token) < 2:
                continue
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vec[digest[0] % self.dim] += 1.0
            vec[digest[1] % self.dim] += 0.5
        norma = sum(x * x for x in vec) ** 0.5
        if norma == 0:
            return vec
        return [x / norma for x in vec]


def get_embeddings(*, dimension: int = EMBEDDING_DIM, offline: bool = False) -> Embeddings:
    """El mismo modelo para indexar y para consultar."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise DimensionMismatchError(
            f"Mismatch de dimensiones: dimension={dimension} es inválida "
            "(tiene que coincidir con el embedding, p.ej. 384)."
        )
    if offline:
        return DeterministicEmbeddings(dim=dimension)
    if dimension != EMBEDDING_DIM:
        raise DimensionMismatchError(
            f"Mismatch de dimensiones: {EMBEDDING_MODEL} es {EMBEDDING_DIM}D "
            f"y se pidió {dimension}D."
        )
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
