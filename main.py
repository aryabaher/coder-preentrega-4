"""Demo: ingesta de las políticas, una consulta híbrida y el reporte de métricas."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from embeddings import EMBEDDING_DIM
from errors import RAGCloudError
from evaluate import cargar_golden, evaluar, imprimir_resumen
from ingesta import BATCH_SIZE, CHUNK_OVERLAP, CHUNK_SIZE, NAMESPACE, TOP_K
from pipeline import preparar_sistema
from rag_system import RAGSystem


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


async def correr(args: argparse.Namespace) -> None:
    inicio = time.perf_counter()
    paquete = await preparar_sistema(
        offline=not args.live,
        namespace=args.namespace,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        k=args.k,
        dimension=args.dimension,
    )
    print(f"tiempo_ingesta_s={time.perf_counter() - inicio:.3f}")
    print(f"chunks={len(paquete['chunks'])} namespace={args.namespace}")

    rag_system: RAGSystem = paquete["rag"]
    resultados = rag_system.obtener_top_k(
        "¿Cuántos días de vacaciones tengo con 6 años de antigüedad?"
    )
    for i, r in enumerate(resultados, 1):
        print(f"{i}. [{r['fuente']}] {r['contenido'][:120]}...")

    reporte = evaluar(rag_system, cargar_golden())
    imprimir_resumen(reporte)


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Demo del RAG híbrido sobre Pinecone.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dimension", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--namespace", default=NAMESPACE)
    args = parser.parse_args()
    try:
        asyncio.run(correr(args))
    except RAGCloudError as exc:
        print(f"Error controlado: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
