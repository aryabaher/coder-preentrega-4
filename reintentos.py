"""Reintentos con backoff. 429 y red sí; 401 y mismatch no."""

from __future__ import annotations

import asyncio
from typing import Callable, Optional, TypeVar

from errors import (
    ClavePineconeError,
    CuotaPineconeError,
    RAGCloudError,
    RedPineconeError,
)

T = TypeVar("T")

MAX_INTENTOS = 3
ESPERA_INICIAL_S = 0.5


def clasificar_error(exc: Exception) -> RAGCloudError:
    if isinstance(exc, RAGCloudError):
        return exc
    texto = str(exc).lower()
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status == 401 or "unauthorized" in texto or "invalid api key" in texto or "api key" in texto:
        return ClavePineconeError(
            f"401/key: clave de Pinecone inválida o ausente. Detalle: {exc}"
        )
    if status == 429 or "quota" in texto or "rate limit" in texto or "resource exhausted" in texto:
        return CuotaPineconeError(
            f"429/cuota: Pinecone rechazó la operación por cuota o rate limit. Detalle: {exc}"
        )
    if (
        status in (408, 504)
        or "timeout" in texto
        or "timed out" in texto
        or "temporarily unavailable" in texto
        or isinstance(exc, (TimeoutError, ConnectionError, OSError))
    ):
        return RedPineconeError(f"red/timeout: no se pudo hablar con Pinecone. Detalle: {exc}")
    return RAGCloudError(f"Error de Pinecone: {exc}")


async def con_reintentos(
    operacion: Callable[[], T],
    *,
    intentos: int = MAX_INTENTOS,
    espera_inicial_s: float = ESPERA_INICIAL_S,
) -> T:
    """Reintenta con backoff ante 429/cuota o red/timeout. 401 y mismatch no."""

    ultimo: Optional[RAGCloudError] = None
    for numero in range(1, intentos + 1):
        try:
            resultado = operacion()
            if asyncio.iscoroutine(resultado):
                resultado = await resultado
            return resultado
        except Exception as exc:  # noqa: BLE001 — se clasifica y se decide el reintento
            clasificado = clasificar_error(exc)
            reintenta = isinstance(clasificado, (CuotaPineconeError, RedPineconeError))
            if reintenta and numero < intentos:
                await asyncio.sleep(espera_inicial_s * (2 ** (numero - 1)))
                ultimo = clasificado
                continue
            raise clasificado from exc
    raise ultimo or RAGCloudError("Reintentos agotados.")
