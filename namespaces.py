"""Namespaces de entorno y de tenant. Un upsert vive en un solo espacio."""

from __future__ import annotations

import re

from errors import EsquemaMetadatosError

NAMESPACES_ENTORNO = ("ns-dev", "ns-staging", "ns-prod")
NAMESPACES_CORPUS = ("politicas-internas",)
ENTORNOS = ("dev", "staging", "prod")
_ORG_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def namespace_de(org_id: str | None = None, entorno: str = "dev") -> str:
    """Un pipeline vive en un solo namespace: entorno o tenant, nunca los dos."""

    if org_id:
        if entorno != "dev":
            raise EsquemaMetadatosError(
                "Un pipeline no mezcla tenant y entorno: usá ns-cliente-<id> "
                "o ns-dev / ns-staging / ns-prod."
            )
        if not isinstance(org_id, str) or not _ORG_RE.fullmatch(org_id):
            raise EsquemaMetadatosError(f"org_id inválido para ns-cliente-<id>: {org_id!r}.")
        return f"ns-cliente-{org_id}"
    if entorno not in ENTORNOS:
        raise EsquemaMetadatosError(
            f"entorno inválido: {entorno!r}. Válidos: {', '.join(ENTORNOS)}."
        )
    return f"ns-{entorno}"


def validar_namespace(namespace: str) -> str:
    if namespace in NAMESPACES_ENTORNO or namespace in NAMESPACES_CORPUS:
        return namespace
    if isinstance(namespace, str) and namespace.startswith("ns-cliente-"):
        org_id = namespace.removeprefix("ns-cliente-")
        if _ORG_RE.fullmatch(org_id):
            return namespace
    raise EsquemaMetadatosError(
        f"namespace inválido: {namespace!r}. Válidos: ns-dev, ns-staging, ns-prod "
        "o ns-cliente-<id>."
    )


def entorno_y_tenant(namespace: str) -> tuple[str, str | None]:
    """ns-dev → (dev, None). ns-cliente-acme → (prod, acme)."""

    validar_namespace(namespace)
    if namespace.startswith("ns-cliente-"):
        return "prod", namespace.removeprefix("ns-cliente-")
    if namespace in NAMESPACES_CORPUS:
        return "dev", None
    return namespace.removeprefix("ns-"), None
