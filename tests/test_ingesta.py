"""Splitter, lotes, metadatos del archivo y CRUD. Sin API real."""

from __future__ import annotations

import pytest

from embeddings import DeterministicEmbeddings
from errors import EsquemaMetadatosError, IngestaError, LoteInvalidoError, SalidaTruncadaError
from indice_local import IndiceLocal
from ingesta import (
    NAMESPACE,
    actualizar_metadatos,
    buscar_vectores,
    cargar_documentos,
    construir_splitter,
    eliminar,
    etiquetar_chunks,
    fragmentar,
    obtener,
    partir_en_lotes,
    preparar_chunks,
    subir_chunks,
    upsert_en_lotes,
    vaciar_namespace,
    validar_batch,
    validar_k,
)
from namespaces import namespace_de
from schemas import MetadatosChunk
from tests.ayudas import ErrorAPI, documento


def test_splitter_usa_el_chunk_size_de_la_llamada():
    splitter = construir_splitter(chunk_size=700, chunk_overlap=90)
    assert splitter._chunk_size == 700
    assert splitter._chunk_overlap == 90
    defecto = construir_splitter()
    assert defecto._chunk_size == 600
    assert defecto._chunk_overlap == 100


def test_chunk_size_fuera_de_rango():
    with pytest.raises(LoteInvalidoError, match=">= 500 y <= 800"):
        construir_splitter(chunk_size=499, chunk_overlap=50)
    with pytest.raises(LoteInvalidoError, match=">= 500 y <= 800"):
        construir_splitter(chunk_size=801, chunk_overlap=50)


def test_chunk_overlap_no_puede_alcanzar_al_size():
    with pytest.raises(IngestaError, match="chunk_overlap"):
        construir_splitter(chunk_size=500, chunk_overlap=500)


def test_batch_size_y_k_fuera_de_limite():
    with pytest.raises(LoteInvalidoError, match="entre 1 y 1000"):
        validar_batch(1001)
    with pytest.raises(LoteInvalidoError, match="entre 1 y 100"):
        validar_k(0)


def test_partir_en_lotes_respeta_el_argumento():
    lotes_default = list(partir_en_lotes(range(250), chunk_size=100))
    assert [len(lote) for lote in lotes_default] == [100, 100, 50]
    lotes = list(partir_en_lotes(range(15), chunk_size=7))
    assert [len(lote) for lote in lotes] == [7, 7, 1]


@pytest.mark.asyncio
async def test_batch_size_llega_al_upsert_y_no_se_clava_en_100():
    indice = IndiceLocal(dimension=4)
    vectores = [
        {"id": f"{i}", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"text": "a"}}
        for i in range(15)
    ]
    total = await upsert_en_lotes(indice, vectores, NAMESPACE, batch_size=7, espera_inicial_s=0)
    assert total == 15
    assert [item["n"] for item in indice.llamadas_upsert] == [7, 7, 1]
    assert {item["batch_size"] for item in indice.llamadas_upsert} == {7}


def test_metadata_sale_del_nombre_de_archivo():
    chunks = etiquetar_chunks(fragmentar([documento("Vacaciones de TechCorp. 14 días.")]))
    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert meta["source"] == "politica_vacaciones.txt"
    assert meta["fuente"] == "politica_vacaciones.txt"
    assert meta["pagina"] == 1
    assert meta["etiquetas"] == ["politica vacaciones"]
    assert meta["categoria"] == "politica vacaciones"
    assert meta["chunk_id"] == 0
    assert meta["text"] == chunks[0].page_content
    MetadatosChunk.model_validate(meta)


def test_carga_txt_markdown_json_y_pdf():
    docs = cargar_documentos()
    nombres = {doc.metadata["source"].replace("\\", "/").split("/")[-1] for doc in docs}
    assert {
        "politica_vacaciones.txt",
        "politica_viajes.md",
        "politica_equipamiento.json",
        "politica_respaldos.pdf",
    } <= nombres
    assert "golden_set.json" not in nombres


def test_schema_drift_en_el_documento():
    doc = documento("cuerpo")
    doc.metadata["date_created"] = "ayer"
    with pytest.raises(EsquemaMetadatosError, match="date_created"):
        etiquetar_chunks([doc])


def test_salida_truncada_no_recorta(monkeypatch):
    monkeypatch.setattr("ingesta.MAX_TEXTO_METADATA", 10)
    with pytest.raises(SalidaTruncadaError, match="Salida truncada"):
        etiquetar_chunks([documento("x" * 40)])


def test_pydantic_rechaza_campo_extra():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MetadatosChunk.model_validate(
            {
                "text": "hola",
                "source": "a.txt",
                "fuente": "a.txt",
                "pagina": 1,
                "etiquetas": ["a"],
                "categoria": "a",
                "chunk_id": 0,
                "date_created": "ayer",
            }
        )


def test_namespaces_de_entorno_y_de_tenant():
    assert namespace_de(entorno="dev") == "ns-dev"
    assert namespace_de(entorno="staging") == "ns-staging"
    assert namespace_de(entorno="prod") == "ns-prod"
    assert namespace_de(org_id="acme") == "ns-cliente-acme"
    with pytest.raises(EsquemaMetadatosError, match="no mezcla"):
        namespace_de(org_id="acme", entorno="prod")


@pytest.mark.asyncio
async def test_subida_guarda_texto_y_filtra_por_namespace():
    indice = IndiceLocal(dimension=8)
    emb = DeterministicEmbeddings(dim=8)
    chunks = preparar_chunks(
        [
            documento("politica de vacaciones del equipo", "politica_vacaciones.txt"),
            documento("TokenUnicoTenantXy solo de acme", "cliente_acme.txt"),
        ],
        chunk_size=500,
        chunk_overlap=40,
    )
    vacaciones = [c for c in chunks if c.metadata["source"] == "politica_vacaciones.txt"]
    tenant = [c for c in chunks if c.metadata["source"] == "cliente_acme.txt"]
    await subir_chunks(vacaciones, emb, namespace=NAMESPACE, index=indice, batch_size=2, dimension=8, espera_inicial_s=0)
    await subir_chunks(
        tenant,
        emb,
        namespace="ns-cliente-acme",
        index=indice,
        batch_size=2,
        dimension=8,
        espera_inicial_s=0,
    )
    guardado = next(iter(indice._datos[NAMESPACE].values()))["metadata"]
    assert guardado["text"]
    assert guardado["source"] == "politica_vacaciones.txt"
    assert guardado["categoria"] == "politica vacaciones"
    respuesta = await buscar_vectores(
        indice,
        emb.embed_query("vacaciones"),
        namespace=NAMESPACE,
        k=5,
        filtro={"source": {"$eq": "politica_vacaciones.txt"}},
    )
    assert respuesta["matches"]
    assert all(m["metadata"]["source"] == "politica_vacaciones.txt" for m in respuesta["matches"])
    assert indice.llamadas_query[-1]["namespace"] == NAMESPACE
    assert indice.llamadas_query[-1]["include_metadata"] is True
    assert indice.conteo("ns-cliente-acme") == 1


@pytest.mark.asyncio
async def test_crud_fetch_update_delete_y_delete_all():
    indice = IndiceLocal(dimension=8)
    emb = DeterministicEmbeddings(dim=8)
    chunks = preparar_chunks([documento("manual para el ciclo de vida")], chunk_size=500, chunk_overlap=40)
    await subir_chunks(chunks, emb, namespace="ns-staging", index=indice, batch_size=10, dimension=8, espera_inicial_s=0)
    chunk_id = str(chunks[0].metadata["chunk_id"])
    fetch = await obtener(indice, chunk_id, "ns-staging")
    assert fetch["metadata"]["source"] == "politica_vacaciones.txt"
    await actualizar_metadatos(indice, chunk_id, {"categoria": "actualizada"}, "ns-staging")
    assert (await obtener(indice, chunk_id, "ns-staging"))["metadata"]["categoria"] == "actualizada"
    await eliminar(indice, [chunk_id], "ns-staging")
    with pytest.raises(IngestaError, match="fetch no encontró"):
        await obtener(indice, chunk_id, "ns-staging")
    extra = preparar_chunks([documento("otro", "otro.txt")], chunk_size=500, chunk_overlap=40)
    await subir_chunks(extra, emb, namespace="ns-cliente-acme", index=indice, batch_size=10, dimension=8, espera_inicial_s=0)
    await vaciar_namespace(indice, "ns-cliente-acme")
    assert indice.conteo("ns-cliente-acme") == 0


@pytest.mark.asyncio
async def test_401_en_upsert_no_reintenta():
    indice = IndiceLocal(dimension=8)

    def upsert(vectors, namespace, batch_size=None, **kwargs):
        upsert.n += 1
        raise ErrorAPI(401, "Unauthorized: invalid API key")

    upsert.n = 0
    indice.upsert = upsert
    with pytest.raises(Exception, match="401/key"):
        await upsert_en_lotes(
            indice,
            [{"id": "a", "values": [0.0] * 8, "metadata": {}}],
            NAMESPACE,
            batch_size=10,
            espera_inicial_s=0,
        )
    assert upsert.n == 1


@pytest.mark.asyncio
async def test_429_en_upsert_reintenta_y_recupera():
    indice = IndiceLocal(dimension=8)
    original = indice.upsert
    estado = {"n": 0}

    def upsert(vectors, namespace, batch_size=None, **kwargs):
        estado["n"] += 1
        if estado["n"] == 1:
            raise ErrorAPI(429, "rate limit / quota exceeded")
        return original(vectors, namespace, batch_size=batch_size, **kwargs)

    indice.upsert = upsert
    total = await upsert_en_lotes(
        indice,
        [{"id": "a", "values": [0.2] * 8, "metadata": {"text": "a"}}],
        NAMESPACE,
        batch_size=10,
        espera_inicial_s=0,
    )
    assert total == 1
    assert estado["n"] == 2
    assert indice.conteo(NAMESPACE) == 1


@pytest.mark.asyncio
async def test_mismatch_de_vector_no_reintenta():
    indice = IndiceLocal(dimension=8)
    with pytest.raises(Exception, match="Mismatch de dimensiones"):
        await upsert_en_lotes(
            indice,
            [{"id": "a", "values": [0.1, 0.2], "metadata": {}}],
            NAMESPACE,
            batch_size=10,
            espera_inicial_s=0,
        )
    assert len(indice.llamadas_upsert) == 1


@pytest.mark.asyncio
async def test_error_no_transitorio_no_reintenta():
    indice = IndiceLocal(dimension=8)

    def upsert(vectors, namespace, batch_size=None, **kwargs):
        upsert.n += 1
        raise RuntimeError("boom")

    upsert.n = 0
    indice.upsert = upsert
    with pytest.raises(Exception, match="Error de Pinecone: boom"):
        await upsert_en_lotes(
            indice,
            [{"id": "a", "values": [0.0] * 8, "metadata": {}}],
            NAMESPACE,
            espera_inicial_s=0,
        )
    assert upsert.n == 1


@pytest.mark.asyncio
async def test_timeout_en_upsert_se_agota():
    indice = IndiceLocal(dimension=8)

    def upsert(vectors, namespace, batch_size=None, **kwargs):
        upsert.n += 1
        raise TimeoutError("Request timed out talking to Pinecone")

    upsert.n = 0
    indice.upsert = upsert
    with pytest.raises(Exception, match="red/timeout"):
        await upsert_en_lotes(
            indice,
            [{"id": "a", "values": [0.0] * 8, "metadata": {}}],
            NAMESPACE,
            espera_inicial_s=0,
        )
    assert upsert.n == 3
