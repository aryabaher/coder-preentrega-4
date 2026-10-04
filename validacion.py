"""Chequeo offline: cadenas de la consigna, errores controlados y métricas."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from config import exigir_pinecone
from embeddings import DeterministicEmbeddings
from errors import (
    ClavePineconeError,
    ConsultaVaciaError,
    EsquemaMetadatosError,
    LoteInvalidoError,
    MetricaMismatchError,
    SalidaTruncadaError,
)
from evaluate import cargar_golden, evaluar, imprimir_resumen
from indice_local import IndiceLocal
from ingesta import construir_splitter, etiquetar_chunks, subir_chunks, upsert_en_lotes
from pipeline import preparar_sistema
from reintentos import clasificar_error
from schemas import ConfigEntorno
from tests.ayudas import ErrorAPI, documento

ROOT = Path(__file__).resolve().parent

CADENAS = [
    "PINECONE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "INDEX_NAME",
    "techcorp-rag-hibrido",
    '[i["name"] for i in pc.list_indexes()]',
    "ServerlessSpec",
    'cloud="aws"',
    'region="us-east-1"',
    'metric="cosine"',
    "dimension=dimension",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
    "OpenAIEmbeddings",
    "1536",
    "DirectoryLoader",
    "TextLoader",
    "RecursiveCharacterTextSplitter",
    "from_tiktoken_encoder",
    "chunk_size=chunk_size",
    "chunk_overlap=chunk_overlap",
    "chunk_size=600",
    "chunk_overlap=100",
    "chunk_size=100",
    "PineconeVectorStore.from_documents",
    "BM25Retriever.from_documents",
    "retriever_bm25.k = 5",
    'search_kwargs={"k": 5, "namespace": NAMESPACE}',
    "EnsembleRetriever",
    "weights=[0.5, 0.5]",
    "class RAGSystem",
    "def obtener_top_k",
    "contenido",
    "categoria",
    "politicas-internas",
    "politica_vacaciones.txt",
    "include_metadata=True",
    "filter=filtro",
    "namespace=namespace",
    "delete_all=True",
    "documento_id_esperado",
    "pregunta",
    "recall@5",
    "precision@5",
    "recall@5_promedio",
    "precision@5_promedio",
    "Precision@5",
    "Recall@5",
    "Precision@k",
    "Recall@k",
    "top-5",
    "ns-dev",
    "ns-staging",
    "ns-prod",
    "ns-cliente-",
    "cosine",
    "1536",
    "metadata[\"text\"]",
    "401/key",
    "429/cuota",
    "red/timeout",
    "Mismatch de dimensiones",
    "Mismatch de métrica",
    "Salida truncada",
    "Schema drift",
]


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


def _ok(nombre: str, cond: bool, detalle: str = "") -> None:
    estado = "OK" if cond else "FALLO"
    extra = f" - {detalle}" if detalle else ""
    print(f"[{estado}] {nombre}{extra}")
    if not cond:
        raise SystemExit(1)


def _fuentes() -> str:
    partes = []
    for ruta in list(ROOT.glob("*.py")) + list((ROOT / "tests").glob("*.py")):
        partes.append(ruta.read_text(encoding="utf-8"))
    partes.append((ROOT / "data" / "golden_set.json").read_text(encoding="utf-8"))
    return "\n".join(partes)


def _chequear_cadenas() -> None:
    texto = _fuentes()
    for cadena in CADENAS:
        _ok(f"cadena {cadena}", cadena in texto)


def _errores() -> None:
    vacio = ConfigEntorno(PINECONE_API_KEY="", INDEX_NAME="techcorp-rag-hibrido")
    try:
        exigir_pinecone(vacio)
    except ClavePineconeError as exc:
        print(f"401 pinecone: {exc}")
        _ok("401 pinecone", str(exc).startswith("401/key: falta PINECONE_API_KEY"))

    cuota = clasificar_error(ErrorAPI(429, "rate limit / quota exceeded"))
    print(f"429: {cuota}")
    _ok("429", str(cuota).startswith("429/cuota:"))
    red = clasificar_error(TimeoutError("Request timed out talking to Pinecone"))
    print(f"red: {red}")
    _ok("red", str(red).startswith("red/timeout:"))

    try:
        IndiceLocal(dimension=0)
    except Exception as exc:  # noqa: BLE001
        print(f"mismatch: {exc}")
        _ok("mismatch dimension", "Mismatch de dimensiones" in str(exc))

    try:
        asyncio.run(_euclid())
    except MetricaMismatchError as exc:
        print(f"metrica: {exc}")
        _ok("mismatch metrica", "Mismatch de métrica" in str(exc))

    malo = documento("hola")
    malo.metadata["date_created"] = "ayer"
    try:
        etiquetar_chunks([malo])
        _ok("schema drift", False, "no lanzó")
    except EsquemaMetadatosError as exc:
        print(f"schema: {exc}")
        _ok("schema drift", "Schema drift" in str(exc) and "date_created" in str(exc))

    try:
        import ingesta

        anterior = ingesta.MAX_TEXTO_METADATA
        ingesta.MAX_TEXTO_METADATA = 5
        try:
            etiquetar_chunks([documento("texto largo de mas")])
        finally:
            ingesta.MAX_TEXTO_METADATA = anterior
    except SalidaTruncadaError as exc:
        print(f"truncado: {exc}")
        _ok("salida truncada", str(exc).startswith("Salida truncada:"))

    try:
        construir_splitter(chunk_size=200, chunk_overlap=10)
    except LoteInvalidoError as exc:
        print(f"chunk: {exc}")
        _ok("chunk_size", ">= 500 y <= 800" in str(exc))


async def _euclid():
    from init_index import asegurar_indice

    await asegurar_indice(object(), "techcorp-rag-hibrido", metric="euclidean")


def _consulta_vacia() -> None:
    from rag_system import construir_retriever_hibrido

    indice = IndiceLocal(dimension=8)
    emb = DeterministicEmbeddings(dim=8)
    chunks = etiquetar_chunks([documento("hay texto de vacaciones en TechCorp")])
    store = asyncio.run(
        subir_chunks(chunks, emb, index=indice, dimension=8, espera_inicial_s=0)
    )
    rag = construir_retriever_hibrido(store, chunks, k=5)[0]
    try:
        rag.obtener_top_k("  ")
    except ConsultaVaciaError as exc:
        print(f"consulta: {exc}")
        _ok("consulta vacia", str(exc).startswith("Consulta vacía:"))


def main() -> None:
    _configure_stdio()
    _chequear_cadenas()
    _errores()
    _consulta_vacia()

    async def _boom():
        indice = IndiceLocal(dimension=4)

        def upsert(vectors, namespace, batch_size=None, **kwargs):
            upsert.n += 1
            raise RuntimeError("boom")

        upsert.n = 0
        indice.upsert = upsert
        try:
            await upsert_en_lotes(
                indice,
                [{"id": "a", "values": [0.1] * 4, "metadata": {"text": "a"}}],
                "ns-dev",
                espera_inicial_s=0,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"no transitorio: {exc}")
            _ok("no transitorio", "Error de Pinecone: boom" in str(exc) and upsert.n == 1)

    asyncio.run(_boom())
    paquete = asyncio.run(preparar_sistema(offline=True, k=5, dimension=1536))
    resultado = evaluar(paquete["rag"], cargar_golden())
    imprimir_resumen(resultado)
    _ok("recall", resultado["recall@5_promedio"] == 1.0)
    _ok("top5", all(len(fila["recuperados"]) == 5 for fila in resultado["detalle"]))
    _ok("precision", abs(resultado["precision@5_promedio"] - 0.2) < 1e-9)
    print("validacion=OK")


if __name__ == "__main__":
    main()
