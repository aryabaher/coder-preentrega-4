"""Aísla los tests de un `.env` real."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("INDEX_NAME", "techcorp-rag-hibrido")
