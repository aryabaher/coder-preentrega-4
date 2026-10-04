"""Políticas de TechCorp: carga, chunks y subida al índice.

El texto original queda en metadata["text"] para no depender de otra base.
La categoría sale del nombre del archivo. El índice de chunk es chunk_id.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ValidationError

from embeddings import EMBEDDING_DIM
from errors import (
    DimensionMismatchError,
    EsquemaMetadatosError,
    IngestaError,
    LoteInvalidoError,
    SalidaTruncadaError,
)
from namespaces import validar_namespace
from reintentos import MAX_INTENTOS, con_reintentos
from schemas import MetadatosChunk

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GOLDEN_PATH = ROOT / "data" / "golden_set.json"

CHUNK_SIZE = 600  # default de llamada: chunk_size=600
CHUNK_OVERLAP = 100  # default de llamada: chunk_overlap=100
BATCH_SIZE = 100
MAX_TEXTO_METADATA = 20000
TOP_K = 5
NAMESPACE = "politicas-internas"

ALIASES_PROHIBIDOS = frozenset({"date_created", "ingest_date", "created_at", "fecha"})

SEPARADORES = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def entero(nombre: str, valor: Any, *, minimo: int, maximo: Optional[int] = None) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int):
        raise LoteInvalidoError(f"{nombre} debe ser un entero. Recibido: {valor!r}.")
    if valor < minimo or (maximo is not None and valor > maximo):
        tope = f" y <= {maximo}" if maximo is not None else ""
        raise LoteInvalidoError(
            f"{nombre} debe ser un entero >= {minimo}{tope}. Recibido: {valor}."
        )
    return valor


def validar_chunk(chunk_size: int, chunk_overlap: int) -> tuple[int, int]:
    chunk_size = entero("chunk_size", chunk_size, minimo=500, maximo=800)
    chunk_overlap = entero("chunk_overlap", chunk_overlap, minimo=0)
    if chunk_overlap >= chunk_size:
        raise IngestaError("chunk_overlap debe ser menor que chunk_size.")
    return chunk_size, chunk_overlap


def validar_batch(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise LoteInvalidoError(
            f"batch_size debe ser un entero entre 1 y 1000 (límite de la API). Recibido: {batch_size!r}."
        )
    if batch_size < 1 or batch_size > 1000:
        raise LoteInvalidoError(
            f"batch_size debe ser un entero entre 1 y 1000 (límite de la API). Recibido: {batch_size}."
        )
    return batch_size


def validar_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > 100:
        raise LoteInvalidoError(f"k debe ser un entero entre 1 y 100. Recibido: {k!r}.")
    return k


def partir_en_lotes(items: Iterable, chunk_size: int = 100) -> Iterator[list]:
    """Parte un iterable en lotes. Default chunk_size=100. El argumento llega al corte."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise LoteInvalidoError(
            f"batch_size debe ser un entero entre 1 y 1000 (límite de la API). Recibido: {chunk_size!r}."
        )
    lote: list = []
    for item in items:
        lote.append(item)
        if len(lote) >= chunk_size:
            yield lote
            lote = []
    if lote:
        yield lote


def construir_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    chunk_size, chunk_overlap = validar_chunk(chunk_size, chunk_overlap)
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(SEPARADORES),
    )


def cargar_documentos(data_dir: str | Path = DATA_DIR) -> List[Document]:
    carpeta = Path(data_dir)
    if not carpeta.is_dir():
        raise IngestaError(f"No existe la carpeta de documentos: {carpeta}")
    loader = DirectoryLoader(
        str(carpeta),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documentos_crudos = loader.load()
    if not documentos_crudos:
        raise IngestaError(f"No hay .txt en {carpeta}.")
    return documentos_crudos


def fragmentar(
    documentos: List[Document],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    splitter = construir_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documentos)


def etiquetar_chunks(chunks: List[Document]) -> List[Document]:
    """Fuente = nombre de archivo. Categoría = ese nombre, sin extensión ni guiones bajos."""

    if not chunks:
        raise IngestaError("El splitter no produjo fragmentos.")
    for i, chunk in enumerate(chunks):
        prohibidos = sorted(ALIASES_PROHIBIDOS & set(chunk.metadata))
        if prohibidos:
            raise EsquemaMetadatosError(
                f"Schema drift: {prohibidos} no pertenece al contrato de metadatos."
            )
        if "source" not in chunk.metadata:
            raise EsquemaMetadatosError("El documento no trae source.")
        texto = (chunk.page_content or "").strip()
        if len(texto) > MAX_TEXTO_METADATA:
            raise SalidaTruncadaError(
                f"Salida truncada: el chunk {i} supera {MAX_TEXTO_METADATA} "
                "caracteres y no se recorta en silencio."
            )
        nombre_archivo = os.path.basename(chunk.metadata["source"])
        categoria = nombre_archivo.replace(".txt", "").replace("_", " ")
        pagina = chunk.metadata.get("pagina", 1)
        try:
            pagina = int(pagina)
        except (TypeError, ValueError) as exc:
            raise EsquemaMetadatosError(f"pagina inválida en {nombre_archivo}.") from exc
        chunk.metadata.clear()
        chunk.metadata["source"] = nombre_archivo
        chunk.metadata["fuente"] = nombre_archivo
        chunk.metadata["pagina"] = pagina
        chunk.metadata["etiquetas"] = [categoria]
        chunk.metadata["categoria"] = categoria
        chunk.metadata["chunk_id"] = i
        chunk.metadata["text"] = texto
        chunk.page_content = texto
        _validar_meta(chunk.metadata)
    return chunks


def _validar_meta(meta: dict) -> MetadatosChunk:
    try:
        return MetadatosChunk.model_validate(meta)
    except ValidationError as exc:
        errores = exc.errors()
        extras = sorted(
            {
                str(err.get("loc", ("",))[0])
                for err in errores
                if err.get("type") == "extra_forbidden"
            }
        )
        if extras:
            raise EsquemaMetadatosError(
                f"Schema drift: {extras} no pertenece al contrato de metadatos."
            ) from exc
        raise EsquemaMetadatosError(f"Esquema de metadatos inválido: {exc}") from exc


def preparar_chunks(
    documentos: List[Document] | None = None,
    *,
    data_dir: str | Path = DATA_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    crudos = documentos if documentos is not None else cargar_documentos(data_dir)
    return etiquetar_chunks(
        fragmentar(crudos, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )


async def upsert_en_lotes(
    index: Any,
    vectores: Iterable,
    namespace: str,
    batch_size: int = BATCH_SIZE,
    *,
    intentos: int = MAX_INTENTOS,
    espera_inicial_s: float = 0.5,
) -> int:
    namespace = validar_namespace(namespace)
    batch_size = validar_batch(batch_size)
    total = 0
    for lote in partir_en_lotes(vectores, chunk_size=batch_size):
        await con_reintentos(
            lambda lote=lote: index.upsert(
                vectors=lote,
                namespace=namespace,
                batch_size=batch_size,
            ),
            intentos=intentos,
            espera_inicial_s=espera_inicial_s,
        )
        total += len(lote)
    return total


async def subir_chunks(
    chunks: List[Document],
    embeddings: Embeddings,
    *,
    namespace: str = NAMESPACE,
    index_name: str | None = None,
    index: Any = None,
    batch_size: int = BATCH_SIZE,
    dimension: int = EMBEDDING_DIM,
    intentos: int = MAX_INTENTOS,
    espera_inicial_s: float = 0.5,
) -> PineconeVectorStore:
    """Sube los chunks. En vivo usa PineconeVectorStore.from_documents. Offline, el índice local."""

    namespace = validar_namespace(namespace)
    batch_size = validar_batch(batch_size)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise DimensionMismatchError(
            f"Mismatch de dimensiones: dimension={dimension} es inválida "
            "(tiene que coincidir con el embedding, p.ej. 1536)."
        )
    ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
    reintento = {"intentos": intentos, "espera_inicial_s": espera_inicial_s}
    if index is None:
        if not index_name:
            raise IngestaError("Falta INDEX_NAME para PineconeVectorStore.from_documents.")

        def _subir():
            return PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=index_name,
                namespace=namespace,
                ids=ids,
                batch_size=batch_size,
            )

        return await con_reintentos(_subir, **reintento)

    store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace,
        text_key="text",
    )
    for lote in partir_en_lotes(chunks, chunk_size=batch_size):
        ids_lote = [str(chunk.metadata["chunk_id"]) for chunk in lote]

        def _lote(lote=lote, ids_lote=ids_lote):
            return store.add_documents(
                lote,
                namespace=namespace,
                ids=ids_lote,
                batch_size=batch_size,
                async_req=True,
            )

        await con_reintentos(_lote, **reintento)
    return store


async def obtener(index: Any, chunk_id: str, namespace: str, **reintento) -> dict:
    namespace = validar_namespace(namespace)
    respuesta = await con_reintentos(
        lambda: index.fetch(ids=[chunk_id], namespace=namespace),
        **reintento,
    )
    vectors = respuesta["vectors"] if isinstance(respuesta, dict) else respuesta.vectors
    if chunk_id not in vectors:
        raise IngestaError(f"fetch no encontró {chunk_id} en {namespace}.")
    item = vectors[chunk_id]
    meta = item["metadata"] if isinstance(item, dict) else item.metadata
    return {"id": chunk_id, "metadata": dict(meta), "namespace": namespace}


async def actualizar_metadatos(index: Any, chunk_id: str, set_metadata: dict, namespace: str, **reintento) -> None:
    namespace = validar_namespace(namespace)
    await con_reintentos(
        lambda: index.update(id=chunk_id, set_metadata=set_metadata, namespace=namespace),
        **reintento,
    )


async def eliminar(index: Any, ids: List[str], namespace: str, **reintento) -> None:
    namespace = validar_namespace(namespace)
    await con_reintentos(lambda: index.delete(ids=ids, namespace=namespace), **reintento)


async def vaciar_namespace(index: Any, namespace: str, **reintento) -> None:
    namespace = validar_namespace(namespace)
    await con_reintentos(
        lambda: index.delete(delete_all=True, namespace=namespace),
        **reintento,
    )


async def buscar_vectores(index: Any, vector: List[float], *, namespace: str, k: int, filtro=None, **reintento):
    namespace = validar_namespace(namespace)
    k = validar_k(k)
    return await con_reintentos(
        lambda: index.query(
            vector=vector,
            top_k=k,
            namespace=namespace,
            filter=filtro,
            include_metadata=True,
        ),
        **reintento,
    )
