"""Índice Serverless: alta, dimensión 1536, métrica y reintentos. Sin red."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from embeddings import EMBEDDING_DIM
from errors import (
    ClavePineconeError,
    CuotaPineconeError,
    DimensionMismatchError,
    MetricaMismatchError,
    RedPineconeError,
)
from init_index import abrir_indice, asegurar_indice
from tests.ayudas import ErrorAPI


class _Lista:
    def __init__(self, nombres):
        self._nombres = list(nombres)

    def __iter__(self):
        return iter({"name": nombre} for nombre in self._nombres)


def _pc(mocker, *, nombres=(), dimension=EMBEDDING_DIM, metric="cosine", ready=True):
    pc = mocker.MagicMock()
    pc.list_indexes.return_value = _Lista(nombres)
    pc.describe_index.return_value = SimpleNamespace(
        dimension=dimension,
        metric=metric,
        status={"ready": ready},
    )
    pc.Index.return_value = mocker.MagicMock(name="index")
    return pc


@pytest.mark.asyncio
async def test_crea_indice_serverless_con_la_dimension_del_modelo(mocker):
    pc = _pc(mocker)
    indice = await asegurar_indice(
        pc,
        "techcorp-rag-hibrido",
        dimension=EMBEDDING_DIM,
        metric="cosine",
        cloud="aws",
        region="us-east-1",
        espera_inicial_s=0,
    )
    kwargs = pc.create_index.call_args.kwargs
    assert kwargs["name"] == "techcorp-rag-hibrido"
    assert kwargs["dimension"] == 1536
    assert kwargs["metric"] == "cosine"
    assert kwargs["spec"].cloud == "aws"
    assert kwargs["spec"].region == "us-east-1"
    pc.Index.assert_called_once_with("techcorp-rag-hibrido")
    assert indice is pc.Index.return_value


@pytest.mark.asyncio
async def test_dimension_distinta_llega_al_create(mocker):
    pc = _pc(mocker, dimension=32)
    await asegurar_indice(
        pc,
        "otro",
        dimension=32,
        cloud="gcp",
        region="us-west1",
        espera_inicial_s=0,
    )
    kwargs = pc.create_index.call_args.kwargs
    assert kwargs["dimension"] == 32
    assert kwargs["spec"].cloud == "gcp"
    assert kwargs["spec"].region == "us-west1"


@pytest.mark.asyncio
async def test_no_recrea_si_existe_y_coincide(mocker):
    pc = _pc(mocker, nombres=["techcorp-rag-hibrido"], dimension=1536, metric="cosine")
    await asegurar_indice(pc, "techcorp-rag-hibrido", dimension=1536, espera_inicial_s=0)
    pc.create_index.assert_not_called()
    pc.Index.assert_called_once_with("techcorp-rag-hibrido")


@pytest.mark.asyncio
async def test_mismatch_de_dimension_no_reintenta(mocker):
    pc = _pc(mocker, nombres=["techcorp-rag-hibrido"], dimension=768, metric="cosine")
    with pytest.raises(DimensionMismatchError, match="768D"):
        await asegurar_indice(pc, "techcorp-rag-hibrido", dimension=1536, espera_inicial_s=0)
    pc.create_index.assert_not_called()
    assert pc.describe_index.call_count == 1


@pytest.mark.asyncio
async def test_mismatch_de_metrica_en_indice_existente(mocker):
    pc = _pc(mocker, nombres=["techcorp-rag-hibrido"], dimension=1536, metric="euclidean")
    with pytest.raises(MetricaMismatchError, match="euclidean"):
        await asegurar_indice(pc, "techcorp-rag-hibrido", dimension=1536, espera_inicial_s=0)
    pc.create_index.assert_not_called()


def test_metrica_euclidean_no_llega_a_crear(mocker):
    pc = _pc(mocker)

    async def correr():
        await asegurar_indice(pc, "techcorp-rag-hibrido", metric="euclidean", espera_inicial_s=0)

    with pytest.raises(MetricaMismatchError, match="cosine"):
        asyncio.run(correr())
    pc.list_indexes.assert_not_called()
    pc.create_index.assert_not_called()


@pytest.mark.asyncio
async def test_dimension_invalida():
    pc = SimpleNamespace()
    with pytest.raises(DimensionMismatchError, match="dimension=0"):
        await asegurar_indice(pc, "techcorp-rag-hibrido", dimension=0)


@pytest.mark.asyncio
async def test_401_en_list_indexes_no_reintenta(mocker):
    pc = _pc(mocker)
    pc.list_indexes.side_effect = ErrorAPI(401, "Unauthorized: invalid API key")
    with pytest.raises(ClavePineconeError, match="401/key"):
        await asegurar_indice(pc, "techcorp-rag-hibrido", espera_inicial_s=0)
    assert pc.list_indexes.call_count == 1
    pc.create_index.assert_not_called()


@pytest.mark.asyncio
async def test_429_reintenta_y_luego_crea(mocker):
    pc = _pc(mocker)
    pc.list_indexes.side_effect = [
        ErrorAPI(429, "rate limit / quota exceeded"),
        _Lista([]),
    ]
    await asegurar_indice(pc, "techcorp-rag-hibrido", dimension=1536, espera_inicial_s=0)
    assert pc.list_indexes.call_count == 2
    pc.create_index.assert_called_once()


@pytest.mark.asyncio
async def test_429_agotado(mocker):
    pc = _pc(mocker)
    pc.list_indexes.side_effect = ErrorAPI(429, "rate limit / quota exceeded")
    with pytest.raises(CuotaPineconeError, match="429/cuota"):
        await asegurar_indice(pc, "techcorp-rag-hibrido", intentos=3, espera_inicial_s=0)
    assert pc.list_indexes.call_count == 3


@pytest.mark.asyncio
async def test_indice_no_ready_es_timeout(mocker):
    pc = _pc(mocker, ready=False)
    with pytest.raises(RedPineconeError, match="no quedó ready"):
        await asegurar_indice(
            pc,
            "techcorp-rag-hibrido",
            timeout_s=0,
            intervalo_s=0,
            espera_inicial_s=0,
        )


@pytest.mark.asyncio
async def test_falta_api_key(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "")
    with pytest.raises(ClavePineconeError, match="falta PINECONE_API_KEY"):
        await abrir_indice(offline=False, index_name="techcorp-rag-hibrido", dimension=1536)


@pytest.mark.asyncio
async def test_offline_no_llama_a_pinecone():
    indice = await abrir_indice(offline=True, index_name="techcorp-rag-hibrido", dimension=32)
    assert indice.dimension == 32
    assert indice.metric == "cosine"


def test_timeout_de_red_se_clasifica():
    from errors import RedPineconeError
    from reintentos import clasificar_error

    exc = clasificar_error(ErrorAPI(504, "Request timed out talking to Pinecone"))
    assert isinstance(exc, RedPineconeError)
    assert "red/timeout" in str(exc)
