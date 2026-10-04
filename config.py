"""Carga `.env` sin pisar variables ya exportadas."""

from __future__ import annotations

import os
from pathlib import Path

from errors import ClavePineconeError
from schemas import ConfigEntorno

ROOT = Path(__file__).resolve().parent


def cargar_dotenv(ruta: str | None = None) -> None:
    """Pasa al entorno las claves de un archivo .env. No pisa variables ya seteadas."""

    archivo = ruta or str(ROOT / ".env")
    try:
        with open(archivo, encoding="utf-8") as fh:
            lineas = fh.readlines()
    except OSError:
        return
    for cruda in lineas:
        linea = cruda.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        clave = clave.strip()
        valor = valor.strip().strip('"').strip("'")
        if clave and clave not in os.environ:
            os.environ[clave] = valor


def leer_config() -> ConfigEntorno:
    cargar_dotenv()
    return ConfigEntorno(
        PINECONE_API_KEY=os.getenv("PINECONE_API_KEY", ""),
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
        ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
        INDEX_NAME=os.getenv("INDEX_NAME", "") or "techcorp-rag-hibrido",
    )


def exigir_pinecone(config: ConfigEntorno | None = None) -> ConfigEntorno:
    cfg = config or leer_config()
    if not cfg.PINECONE_API_KEY.strip():
        raise ClavePineconeError(
            "401/key: falta PINECONE_API_KEY en las variables de entorno."
        )
    return cfg

