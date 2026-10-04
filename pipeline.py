"""Arma el índice, ingesta las políticas y devuelve el RAGSystem híbrido."""

from __future__ import annotations

from typing import Any, Dict, Optional

from config import exigir_pinecone, leer_config
from embeddings import EMBEDDING_DIM, get_embeddings
from ingesta import (
    BATCH_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    NAMESPACE,
    TOP_K,
    preparar_chunks,
    subir_chunks,
)
from init_index import abrir_indice
from rag_system import WEIGHTS, construir_demo, construir_retriever_hibrido


async def preparar_sistema(
    *,
    offline: bool = True,
    namespace: str = NAMESPACE,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size: int = BATCH_SIZE,
    k: int = TOP_K,
    dimension: int = EMBEDDING_DIM,
    index_name: Optional[str] = None,
    data_dir=None,
    weights=None,
) -> Dict[str, Any]:
    config = leer_config()
    if not offline:
        exigir_pinecone(config)
    nombre = index_name or config.INDEX_NAME
    indice = await abrir_indice(
        offline=offline,
        index_name=nombre,
        dimension=dimension,
        metric="cosine",
        cloud="aws",
        region="us-east-1",
    )
    embeddings = get_embeddings(dimension=dimension, offline=offline)
    chunks = preparar_chunks(
        data_dir=data_dir or DATA_DIR,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    vectorstore = await subir_chunks(
        chunks,
        embeddings,
        namespace=namespace,
        index_name=None if offline else nombre,
        index=indice if offline else None,
        batch_size=batch_size,
        dimension=dimension,
    )
    if k == TOP_K and namespace == NAMESPACE and weights is None:
        rag = construir_demo(vectorstore, chunks)
    else:
        rag, *_resto = construir_retriever_hibrido(
            vectorstore,
            chunks,
            k=k,
            namespace=namespace,
            weights=weights or list(WEIGHTS),
        )
    return {
        "rag": rag,
        "index": indice,
        "embeddings": embeddings,
        "chunks": chunks,
        "vectorstore": vectorstore,
        "index_name": nombre,
    }
