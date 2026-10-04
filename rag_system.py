"""RAGSystem: EnsembleRetriever con BM25Retriever y PineconeVectorStore.

El k de la llamada llega a BM25Retriever.k, a search_kwargs y al corte del top-k.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from errors import ConsultaVaciaError, RAGCloudError
from ingesta import NAMESPACE, validar_k
from reintentos import clasificar_error

TOP_K = 5
WEIGHTS = [0.5, 0.5]


class RAGSystem:
    """Encapsula el EnsembleRetriever y expone un método simple para obtener el top-k."""

    def __init__(self, retriever, k: int = 5):
        self.retriever = retriever
        self.k = validar_k(k)

    def obtener_top_k(self, query: str) -> List[Dict]:
        texto = (query or "").strip()
        if not texto:
            raise ConsultaVaciaError("Consulta vacía: no hay texto para recuperar.")
        try:
            self._propagar_k()
            docs = self.retriever.invoke(texto)[: self.k]
            return [
                {
                    "contenido": d.page_content,
                    "fuente": d.metadata.get("source", "desconocida"),
                    "categoria": d.metadata.get("categoria", "desconocida"),
                }
                for d in docs
            ]
        except RAGCloudError:
            raise
        except Exception as exc:  # noqa: BLE001 — Pinecone no comparte un solo tipo de error
            raise clasificar_error(exc) from exc

    def _propagar_k(self) -> None:
        piezas = getattr(self.retriever, "retrievers", None)
        if not piezas:
            return
        for pieza in piezas:
            if hasattr(pieza, "k"):
                pieza.k = self.k
            kwargs = getattr(pieza, "search_kwargs", None)
            if isinstance(kwargs, dict):
                kwargs["k"] = self.k


def construir_retriever_hibrido(
    vectorstore,
    chunks: List[Document],
    *,
    k: int = TOP_K,
    namespace: str = NAMESPACE,
    weights: Optional[List[float]] = None,
):
    """BM25 + Pinecone. `k` y `namespace` llegan a los dos retrievers."""

    k = validar_k(k)
    pesos = list(weights or WEIGHTS)
    if len(pesos) != 2 or any(peso <= 0 for peso in pesos):
        raise RAGCloudError("weights debe tener dos valores > 0 (BM25 y vectorial).")
    retriever_bm25 = BM25Retriever.from_documents(chunks)
    retriever_bm25.k = k
    retriever_vectorial = vectorstore.as_retriever(
        search_kwargs={"k": k, "namespace": namespace},
    )
    retriever_hibrido = EnsembleRetriever(
        retrievers=[retriever_bm25, retriever_vectorial],
        weights=pesos,
    )
    return RAGSystem(retriever_hibrido, k=k), retriever_bm25, retriever_vectorial, retriever_hibrido


def construir_demo(vectorstore, chunks: List[Document]):
    """El armado por defecto del corpus: k=5 y el namespace de políticas."""

    retriever_bm25 = BM25Retriever.from_documents(chunks)
    retriever_bm25.k = 5
    retriever_vectorial = vectorstore.as_retriever(
        search_kwargs={"k": 5, "namespace": NAMESPACE},
    )
    retriever_hibrido = EnsembleRetriever(
        retrievers=[retriever_bm25, retriever_vectorial],
        weights=[0.5, 0.5],
    )
    rag_system = RAGSystem(retriever_hibrido, k=5)
    return rag_system
