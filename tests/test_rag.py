"""RAGSystem.obtener_top_k combina BM25 y Pinecone. El k llega a los dos."""

from __future__ import annotations

import pytest

from embeddings import EMBEDDING_DIM, EMBEDDING_MODEL, DeterministicEmbeddings, get_embeddings
from errors import ConsultaVaciaError, DimensionMismatchError
from indice_local import IndiceLocal
from ingesta import NAMESPACE, preparar_chunks, subir_chunks
from rag_system import RAGSystem, construir_retriever_hibrido
from tests.ayudas import documento


async def _corpus():
    indice = IndiceLocal(dimension=8)
    emb = DeterministicEmbeddings(dim=8)
    base = [
        documento(f"Manual {i} de operaciones internas del servicio {i}.", f"manual_{i}.txt")
        for i in range(5)
    ]
    base.append(documento("El identificador TokenRaroZz9 aparece solo en soporte.", "soporte.txt"))
    chunks = preparar_chunks(base, chunk_size=500, chunk_overlap=40)
    store = await subir_chunks(
        chunks, emb, namespace=NAMESPACE, index=indice, batch_size=2, dimension=8, espera_inicial_s=0
    )
    return indice, store, chunks


@pytest.mark.asyncio
async def test_k_de_la_llamada_llega_al_bm25_y_a_pinecone():
    indice, store, chunks = await _corpus()
    rag, bm25, vectorial, ensemble = construir_retriever_hibrido(store, chunks, k=3, namespace=NAMESPACE)
    assert isinstance(rag, RAGSystem)
    assert bm25.k == 3
    assert vectorial.search_kwargs["k"] == 3
    assert vectorial.search_kwargs["namespace"] == NAMESPACE
    assert ensemble.__class__.__name__ == "EnsembleRetriever"
    assert bm25.__class__.__name__ == "BM25Retriever"
    assert store.__class__.__name__ == "PineconeVectorStore"
    hits = rag.obtener_top_k("servicio")
    assert len(hits) == 3
    assert bm25.k == 3
    assert vectorial.search_kwargs["k"] == 3
    assert indice.llamadas_query[-1]["top_k"] == 3
    assert indice.llamadas_query[-1]["include_metadata"] is True
    for hit in hits:
        assert {"contenido", "fuente", "categoria"} <= set(hit)


@pytest.mark.asyncio
async def test_token_raro_queda_primero_y_el_default_es_top_5():
    indice, store, chunks = await _corpus()
    rag = construir_retriever_hibrido(store, chunks, k=5, namespace=NAMESPACE)[0]
    hits = rag.obtener_top_k("TokenRaroZz9")
    assert len(hits) == 5
    assert hits[0]["fuente"] == "soporte.txt"
    assert hits[0]["categoria"] == "soporte"


@pytest.mark.asyncio
async def test_namespace_no_mezcla_al_tenant():
    indice, store, chunks = await _corpus()
    tenant_chunks = preparar_chunks(
        [documento("TokenUnicoTenantXy secreto de acme", "cliente_acme.txt")],
        chunk_size=500,
        chunk_overlap=40,
    )
    await subir_chunks(
        tenant_chunks,
        DeterministicEmbeddings(dim=8),
        namespace="ns-cliente-acme",
        index=indice,
        batch_size=2,
        dimension=8,
        espera_inicial_s=0,
    )
    rag, *_ = construir_retriever_hibrido(store, chunks, k=5, namespace=NAMESPACE)
    dev = rag.obtener_top_k("TokenUnicoTenantXy")
    assert all(hit["fuente"] != "cliente_acme.txt" for hit in dev)


def test_consulta_vacia():
    indice = IndiceLocal(dimension=8)
    emb = DeterministicEmbeddings(dim=8)
    chunks = preparar_chunks([documento("hay texto")], chunk_size=500, chunk_overlap=40)

    async def subir():
        return await subir_chunks(chunks, emb, index=indice, dimension=8, espera_inicial_s=0)

    import asyncio

    store = asyncio.run(subir())
    rag = construir_retriever_hibrido(store, chunks, k=5)[0]
    with pytest.raises(ConsultaVaciaError, match="Consulta vacía"):
        rag.obtener_top_k("   ")


def test_weights_llegan_al_ensemble():
    indice = IndiceLocal(dimension=8)
    emb = DeterministicEmbeddings(dim=8)
    chunks = preparar_chunks([documento("hay texto")], chunk_size=500, chunk_overlap=40)

    async def subir():
        return await subir_chunks(chunks, emb, index=indice, dimension=8, espera_inicial_s=0)

    import asyncio

    store = asyncio.run(subir())
    _rag, _bm25, _vec, ensemble = construir_retriever_hibrido(
        store, chunks, weights=[0.2, 0.8]
    )
    assert list(ensemble.weights) == [0.2, 0.8]


def test_embeddings_openai_recibe_modelo_y_dimension(mocker):
    mock = mocker.patch("langchain_openai.OpenAIEmbeddings")
    get_embeddings(dimension=EMBEDDING_DIM, offline=False, api_key="sk-test")
    kwargs = mock.call_args.kwargs
    assert kwargs["model"] == EMBEDDING_MODEL == "text-embedding-3-small"
    assert kwargs["dimensions"] == 1536
    assert kwargs["api_key"] == "sk-test"


def test_embeddings_offline_usa_la_dimension_pedida():
    emb = get_embeddings(dimension=32, offline=True)
    assert len(emb.embed_query("TechCorp")) == 32


def test_openai_sin_clave():
    from errors import ClavePineconeError

    with pytest.raises(ClavePineconeError, match="falta OPENAI_API_KEY"):
        get_embeddings(dimension=1536, offline=False, api_key="")


def test_openai_rechaza_otro_ancho():
    with pytest.raises(DimensionMismatchError, match="1536D"):
        get_embeddings(dimension=384, offline=False, api_key="sk-test")
