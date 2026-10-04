"""evaluate.py: Precision@5 y Recall@5 sobre el golden set de TechCorp.

Cada pregunta tiene un solo documento_id_esperado. Con un único documento
relevante, Recall@5 solo puede ser 0 o 1. Precision@5 es la proporción de
fragmentos recuperados cuya fuente es esa.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

from pydantic import ValidationError

from embeddings import EMBEDDING_DIM
from errors import IngestaError, RAGCloudError
from ingesta import BATCH_SIZE, CHUNK_OVERLAP, CHUNK_SIZE, GOLDEN_PATH, NAMESPACE, TOP_K
from pipeline import preparar_sistema
from rag_system import RAGSystem
from schemas import ItemGolden


GOLDEN_SET = [
    {
        "pregunta": "¿Cuántos días de vacaciones corresponden a un empleado con 6 años de antigüedad?",
        "documento_id_esperado": "politica_vacaciones.txt",
    },
    {
        "pregunta": "¿Cuántos días de trabajo remoto por semana tiene el esquema estándar de TechCorp?",
        "documento_id_esperado": "politica_teletrabajo.txt",
    },
    {
        "pregunta": "¿Cada cuánto deben renovarse las contraseñas corporativas?",
        "documento_id_esperado": "politica_seguridad_informatica.txt",
    },
    {
        "pregunta": "¿Cuánto dura el proceso de onboarding en TechCorp?",
        "documento_id_esperado": "onboarding_nuevos_empleados.txt",
    },
    {
        "pregunta": "¿Qué esquema de trabajo tienen las áreas de Soporte Técnico Nivel 1 y Recepción?",
        "documento_id_esperado": "politica_teletrabajo.txt",
    },
]


def precision_at_k(recuperados: List[str], esperado: str, k: int) -> float:
    """Precision@k sobre la lista ya cortada. El denominador es cuántos volvieron."""

    top = list(recuperados)[:k]
    if not top:
        return 0.0
    coincidencias = sum(1 for fuente in top if fuente == esperado)
    return coincidencias / len(top)


def recall_at_k(recuperados: List[str], esperado: str, k: int) -> float:
    """Recall@k = 1 si el documento esperado está en el top-k."""

    return 1.0 if esperado in list(recuperados)[:k] else 0.0


def cargar_golden(ruta: str | Path = GOLDEN_PATH) -> List[ItemGolden]:
    path = Path(ruta)
    if path.is_file():
        try:
            crudo = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IngestaError(f"No se pudo leer el golden set: {exc}") from exc
    else:
        crudo = GOLDEN_SET
    if not isinstance(crudo, list):
        raise IngestaError("El golden set tiene que ser una lista de preguntas.")
    try:
        items = [ItemGolden.model_validate(fila) for fila in crudo]
    except ValidationError as exc:
        raise IngestaError(f"Golden set inválido: {exc}") from exc
    if len(items) < 5:
        raise IngestaError("El golden set necesita al menos 5 preguntas.")
    return items


def evaluar(rag_system: RAGSystem, golden_set: List[Dict]) -> Dict:
    resultados_por_pregunta = []

    for caso in golden_set:
        pregunta = caso["pregunta"] if isinstance(caso, dict) else caso.pregunta
        esperado = caso["documento_id_esperado"] if isinstance(caso, dict) else caso.documento_id_esperado
        top_k = rag_system.obtener_top_k(pregunta)
        fuentes_recuperadas = [r["fuente"] for r in top_k]

        recall = 1.0 if esperado in fuentes_recuperadas else 0.0
        coincidencias = sum(1 for fuente in fuentes_recuperadas if fuente == esperado)
        precision = coincidencias / len(top_k) if top_k else 0.0

        resultados_por_pregunta.append(
            {
                "pregunta": pregunta,
                "esperado": esperado,
                "recuperados": fuentes_recuperadas,
                "recall@5": recall,
                "precision@5": precision,
            }
        )

    n = len(resultados_por_pregunta)
    recall_promedio = sum(r["recall@5"] for r in resultados_por_pregunta) / n
    precision_promedio = sum(r["precision@5"] for r in resultados_por_pregunta) / n
    return {
        "detalle": resultados_por_pregunta,
        "recall@5_promedio": recall_promedio,
        "precision@5_promedio": precision_promedio,
        "k": rag_system.k,
    }


def imprimir_resumen(reporte: Dict) -> None:
    print("=" * 80)
    for r in reporte["detalle"]:
        estado = "OK" if r["recall@5"] == 1.0 else "FALLO"
        print(f"{estado} {r['pregunta']}")
        print(f"   Esperado: {r['esperado']} | Recuperados: {r['recuperados']}")
        print(f"   Recall@5: {r['recall@5']:.0%} | Precision@5: {r['precision@5']:.0%}")
    print("=" * 80)
    print(f"RECALL@5 PROMEDIO:    {reporte['recall@5_promedio']:.1%}")
    print(f"PRECISION@5 PROMEDIO: {reporte['precision@5_promedio']:.1%}")
    print(f"Precision@k: {reporte['precision@5_promedio']:.4f}")
    print(f"Recall@k: {reporte['recall@5_promedio']:.4f}")
    if reporte.get("k", 5) == 5:
        print(f"Precision@5: {reporte['precision@5_promedio']:.4f}")
        print(f"Recall@5: {reporte['recall@5_promedio']:.4f}")
        print("top-5 documentos por pregunta")


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Precision@5 y Recall@5 del recuperador híbrido.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dimension", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--namespace", default=NAMESPACE)
    args = parser.parse_args()
    try:
        paquete = asyncio.run(
            preparar_sistema(
                offline=not args.live,
                namespace=args.namespace,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
                k=args.k,
                dimension=args.dimension,
            )
        )
        resultado = evaluar(paquete["rag"], cargar_golden())
        imprimir_resumen(resultado)
    except RAGCloudError as exc:
        print(f"Error controlado: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
