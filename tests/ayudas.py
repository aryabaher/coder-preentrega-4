"""Documentos chicos y errores de API para los tests. No pegan a Pinecone."""

from __future__ import annotations

from langchain_core.documents import Document


class ErrorAPI(Exception):
    def __init__(self, status: int, mensaje: str) -> None:
        super().__init__(mensaje)
        self.status = status


def documento(
    texto: str = "Texto tecnico de TechCorp sobre vacaciones y antigüedad.",
    nombre: str = "politica_vacaciones.txt",
) -> Document:
    return Document(page_content=texto, metadata={"source": nombre})
