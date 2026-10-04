"""Precision@5, Recall@5 y el golden set de las políticas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluate import GOLDEN_SET, cargar_golden, evaluar, precision_at_k, recall_at_k
from pipeline import preparar_sistema
from schemas import ItemGolden


def test_precision_y_recall_at_k():
    recuperados = ["politica_vacaciones.txt", "politica_teletrabajo.txt", "politica_vacaciones.txt"]
    assert precision_at_k(recuperados, "politica_vacaciones.txt", 5) == pytest.approx(2 / 3)
    assert recall_at_k(recuperados, "politica_vacaciones.txt", 5) == 1.0
    assert recall_at_k(["politica_teletrabajo.txt"], "politica_vacaciones.txt", 5) == 0.0
    assert precision_at_k([], "politica_vacaciones.txt", 5) == 0.0


def test_golden_set_tiene_cinco_preguntas():
    items = cargar_golden()
    assert len(items) == 5
    assert [item.documento_id_esperado for item in items] == [
        fila["documento_id_esperado"] for fila in GOLDEN_SET
    ]
    assert items[0].documento_id_esperado == "politica_vacaciones.txt"
    assert items[4].documento_id_esperado == "politica_teletrabajo.txt"


def test_item_golden_rechaza_campo_extra():
    with pytest.raises(ValidationError):
        ItemGolden.model_validate(
            {"pregunta": "¿?", "documento_id_esperado": "politica_vacaciones.txt", "pista": "no"}
        )


class _RagFijo:
    def __init__(self):
        self.k = 5
        self.preguntas = []

    def obtener_top_k(self, query: str):
        self.preguntas.append(query)
        return [{"contenido": "texto", "fuente": "politica_vacaciones.txt", "categoria": "x"}]


def test_evaluar_usa_obtener_top_k_y_promedia():
    rag = _RagFijo()
    reporte = evaluar(rag, GOLDEN_SET)
    assert len(rag.preguntas) == 5
    assert reporte["recall@5_promedio"] == pytest.approx(0.2)
    assert reporte["precision@5_promedio"] == pytest.approx(0.2)
    assert set(reporte["detalle"][0]) >= {"recall@5", "precision@5", "esperado", "recuperados"}


@pytest.mark.asyncio
async def test_corpus_real_recupera_el_documento_esperado():
    paquete = await preparar_sistema(offline=True, k=5, dimension=1536)
    reporte = evaluar(paquete["rag"], GOLDEN_SET)
    assert reporte["recall@5_promedio"] == 1.0
    assert reporte["precision@5_promedio"] == pytest.approx(0.2)
    assert all(len(fila["recuperados"]) == 5 for fila in reporte["detalle"])
    assert all(fila["recall@5"] == 1.0 for fila in reporte["detalle"])
    assert reporte["detalle"][0]["recuperados"][0] == "politica_vacaciones.txt"
