"""Configuración: `.env` no pisa el entorno y el nombre del índice se valida."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from config import cargar_dotenv, leer_config
from schemas import ConfigEntorno


def test_dotenv_carga_y_no_pisa(tmp_path, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY_TEST", raising=False)
    archivo = tmp_path / ".env"
    archivo.write_text("PINECONE_API_KEY_TEST=desde-archivo\n", encoding="utf-8")
    cargar_dotenv(str(archivo))
    assert os.getenv("PINECONE_API_KEY_TEST") == "desde-archivo"
    monkeypatch.setenv("PINECONE_API_KEY_TEST", "ya-seteada")
    archivo.write_text("PINECONE_API_KEY_TEST=otra\n", encoding="utf-8")
    cargar_dotenv(str(archivo))
    assert os.getenv("PINECONE_API_KEY_TEST") == "ya-seteada"


def test_leer_config_toma_el_entorno(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    monkeypatch.setenv("INDEX_NAME", "techcorp-rag-hibrido")
    cfg = leer_config()
    assert cfg.PINECONE_API_KEY == "pk"
    assert cfg.INDEX_NAME == "techcorp-rag-hibrido"


def test_index_name_vacio():
    with pytest.raises(ValidationError, match="INDEX_NAME"):
        ConfigEntorno.model_validate({"PINECONE_API_KEY": "", "INDEX_NAME": "  "})
