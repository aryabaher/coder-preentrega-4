"""Inicializa el índice Pinecone Serverless si no existe y aborta si no calza.

Contrato de namespaces
- Entornos: `ns-dev`, `ns-staging`, `ns-prod`.
- Tenants: `ns-cliente-<id>`.

Métrica vs embedding
- OpenAI text-embedding-3-small → dimensión 1536 y métrica cosine.
- Si el índice ya existe con otro ancho (512, 768, 384) o con otra métrica, se aborta. No se reintenta.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any, Optional

from pinecone import Pinecone, ServerlessSpec

from config import exigir_pinecone, leer_config
from embeddings import EMBEDDING_DIM
from errors import (
    DimensionMismatchError,
    MetricaMismatchError,
    RAGCloudError,
    RedPineconeError,
)
from indice_local import IndiceLocal
from reintentos import MAX_INTENTOS, con_reintentos

TIMEOUT_INDICE_S = 120
INTERVALO_POLL_S = 2


def _dimension_de(desc: Any) -> Optional[int]:
    dim = getattr(desc, "dimension", None)
    if dim is None and isinstance(desc, dict):
        dim = desc.get("dimension")
    return int(dim) if dim is not None else None


def _metrica_de(desc: Any) -> Optional[str]:
    metric = getattr(desc, "metric", None)
    if metric is None and isinstance(desc, dict):
        metric = desc.get("metric")
    if metric is None:
        return None
    return str(getattr(metric, "value", metric))


def _indice_listo(desc: Any) -> bool:
    status = getattr(desc, "status", None)
    if status is None and isinstance(desc, dict):
        status = desc.get("status")
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


def _verificar_compatible(desc: Any, index_name: str, dimension: int, metric: str) -> None:
    existente = _dimension_de(desc)
    if existente is not None and existente != dimension:
        raise DimensionMismatchError(
            f"Mismatch de dimensiones: el índice {index_name} es {existente}D "
            f"y se pidió {dimension}D."
        )
    actual = _metrica_de(desc)
    if actual is not None and actual != metric:
        raise MetricaMismatchError(
            f"Mismatch de métrica: el índice {index_name} es {actual} y se pidió {metric}."
        )


async def _esperar_indice_listo(
    pc: Pinecone,
    index_name: str,
    *,
    timeout_s: int = TIMEOUT_INDICE_S,
    intervalo_s: float = INTERVALO_POLL_S,
    intentos: int = MAX_INTENTOS,
    espera_inicial_s: float = 0.5,
) -> None:
    inicio = time.perf_counter()
    ultimo: Optional[Exception] = None
    while time.perf_counter() - inicio < timeout_s:
        try:
            desc = await con_reintentos(
                lambda: pc.describe_index(index_name),
                intentos=intentos,
                espera_inicial_s=espera_inicial_s,
            )
            if _indice_listo(desc):
                return
        except RAGCloudError as exc:
            ultimo = exc
        await asyncio.sleep(intervalo_s)
    detalle = f" Último error: {ultimo}" if ultimo else ""
    raise RedPineconeError(
        f"red/timeout: el índice {index_name} no quedó ready en {timeout_s}s.{detalle}"
    )


async def asegurar_indice(
    pc: Pinecone,
    index_name: str,
    dimension: int = EMBEDDING_DIM,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
    *,
    timeout_s: int = TIMEOUT_INDICE_S,
    intervalo_s: float = INTERVALO_POLL_S,
    intentos: int = MAX_INTENTOS,
    espera_inicial_s: float = 0.5,
):
    """Crea el índice Serverless si no está. Si está, exige la misma dimensión y métrica."""

    if not index_name or not str(index_name).strip():
        raise RAGCloudError("INDEX_NAME vacío.")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise DimensionMismatchError(
            f"Mismatch de dimensiones: dimension={dimension} es inválida "
            "(tiene que coincidir con el embedding, p.ej. 1536)."
        )
    if metric != "cosine":
        raise MetricaMismatchError(
            f"Mismatch de métrica: se pidió {metric} y text-embedding-3-small usa cosine."
        )
    reintento = {"intentos": intentos, "espera_inicial_s": espera_inicial_s}
    nombres = await con_reintentos(
        lambda: [i["name"] for i in pc.list_indexes()],
        **reintento,
    )
    if index_name not in nombres:
        await con_reintentos(
            lambda: pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            ),
            **reintento,
        )
        await _esperar_indice_listo(
            pc,
            index_name,
            timeout_s=timeout_s,
            intervalo_s=intervalo_s,
            intentos=intentos,
            espera_inicial_s=espera_inicial_s,
        )
    else:
        desc = await con_reintentos(lambda: pc.describe_index(index_name), **reintento)
        _verificar_compatible(desc, index_name, dimension, metric)
    return pc.Index(index_name)


async def abrir_indice(
    *,
    offline: bool,
    index_name: str,
    dimension: int = EMBEDDING_DIM,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
    timeout_s: int = TIMEOUT_INDICE_S,
) -> Any:
    if offline:
        if metric != "cosine":
            raise MetricaMismatchError(
                f"Mismatch de métrica: se pidió {metric} y text-embedding-3-small usa cosine."
            )
        return IndiceLocal(dimension=dimension, metric=metric)
    config = exigir_pinecone()
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    return await asegurar_indice(
        pc,
        index_name,
        dimension=dimension,
        metric=metric,
        cloud=cloud,
        region=region,
        timeout_s=timeout_s,
    )


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Crea el índice Pinecone Serverless si falta.")
    parser.add_argument("--live", action="store_true", help="Habla con Pinecone. Sin esto, índice local.")
    parser.add_argument("--dimension", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--index-name", default=None)
    args = parser.parse_args()
    config = leer_config()
    nombre = args.index_name or config.INDEX_NAME
    try:
        if args.live:
            asyncio.run(
                abrir_indice(
                    offline=False,
                    index_name=nombre,
                    dimension=args.dimension,
                    metric="cosine",
                    cloud="aws",
                    region="us-east-1",
                )
            )
            print(
                f"índice={nombre} dimension={args.dimension} metric=cosine "
                "spec=ServerlessSpec cloud=aws region=us-east-1"
            )
        else:
            asyncio.run(
                abrir_indice(
                    offline=True,
                    index_name=nombre,
                    dimension=args.dimension,
                    metric="cosine",
                    cloud="aws",
                    region="us-east-1",
                )
            )
            print(
                f"indice_local={nombre} dimension={args.dimension} metric=cosine "
                "spec=ServerlessSpec cloud=aws region=us-east-1"
            )
            print("Modo offline: no se creó un índice en Pinecone. Repetí con --live y las API keys.")
    except RAGCloudError as exc:
        print(f"Error controlado: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
