"""Índice en memoria con la misma forma que el Index de Pinecone.

Sirve para correr ingesta, RAG híbrido y evaluate.py sin API key.
El modo live usa el Index real; esta clase no lo reemplaza ahí.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from errors import DimensionMismatchError, EsquemaMetadatosError
from namespaces import validar_namespace


def _coseno(a: List[float], b: List[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    if denom == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def _como_dict(vector: Any) -> Dict[str, Any]:
    if isinstance(vector, dict):
        return {
            "id": vector["id"],
            "values": list(vector["values"]),
            "metadata": dict(vector.get("metadata") or {}),
        }
    vid, values, metadata = vector
    return {"id": vid, "values": list(values), "metadata": dict(metadata or {})}


def _pasa_filtro(metadata: Dict[str, Any], filtro: Optional[Dict]) -> bool:
    if not filtro:
        return True
    for clave, condicion in filtro.items():
        valor = metadata.get(clave)
        if not isinstance(condicion, dict):
            if isinstance(valor, list):
                if condicion not in valor:
                    return False
            elif valor != condicion:
                return False
            continue
        if "$eq" in condicion:
            esperado = condicion["$eq"]
            if isinstance(valor, list):
                if esperado not in valor:
                    return False
            elif valor != esperado:
                return False
        if "$in" in condicion:
            opciones = condicion["$in"]
            if isinstance(valor, list):
                if not any(item in opciones for item in valor):
                    return False
            elif valor not in opciones:
                return False
        if "$gt" in condicion:
            umbral = condicion["$gt"]
            if isinstance(valor, bool) or not isinstance(valor, (int, float)) or valor <= umbral:
                return False
    return True


def _copiar_meta(metadata: Dict[str, Any]) -> Dict[str, Any]:
    copia: Dict[str, Any] = {}
    for clave, valor in metadata.items():
        copia[clave] = list(valor) if isinstance(valor, list) else valor
    return copia


class _RespuestaAsync:
    """Misma forma que el upsert async del SDK: hay que llamar a get()."""

    def __init__(self, valor: Any) -> None:
        self._valor = valor

    def get(self) -> Any:
        return self._valor


class IndiceLocal:
    """Upsert, query, fetch, update y delete por namespace. Métrica cosine."""

    def __init__(self, dimension: int, metric: str = "cosine") -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise DimensionMismatchError(
                f"Mismatch de dimensiones: dimension={dimension} es inválida "
                "(tiene que coincidir con el embedding, p.ej. 1536)."
            )
        self.dimension = dimension
        self.metric = metric
        self.config = SimpleNamespace(host="indice-local", api_key="offline")
        self._datos: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.llamadas_upsert: List[Dict[str, Any]] = []
        self.llamadas_query: List[Dict[str, Any]] = []

    def conteo(self, namespace: str) -> int:
        return len(self._datos.get(namespace, {}))

    def upsert(self, vectors, namespace: str, batch_size: int | None = None, async_req: bool = False, **kwargs):
        validar_namespace(namespace)
        lote = [_como_dict(item) for item in list(vectors)]
        self.llamadas_upsert.append(
            {"namespace": namespace, "n": len(lote), "batch_size": batch_size, "async_req": async_req}
        )
        espacio = self._datos.setdefault(namespace, {})
        for item in lote:
            if len(item["values"]) != self.dimension:
                raise DimensionMismatchError(
                    f"Mismatch de dimensiones: el vector es {len(item['values'])}D "
                    f"y el índice es {self.dimension}D."
                )
            espacio[item["id"]] = {
                "values": item["values"],
                "metadata": _copiar_meta(item["metadata"]),
            }
        respuesta = SimpleNamespace(upserted_count=len(lote))
        if async_req:
            return _RespuestaAsync(respuesta)
        return respuesta

    def query(
        self,
        vector,
        top_k: int = 5,
        namespace: str | None = None,
        filter=None,
        include_metadata: bool = True,
        **kwargs,
    ):
        if not namespace:
            raise EsquemaMetadatosError("La consulta tiene que indicar namespace=.")
        validar_namespace(namespace)
        if len(vector) != self.dimension:
            raise DimensionMismatchError(
                f"Mismatch de dimensiones: la consulta es {len(vector)}D "
                f"y el índice es {self.dimension}D."
            )
        self.llamadas_query.append(
            {
                "namespace": namespace,
                "top_k": top_k,
                "filter": filter,
                "include_metadata": include_metadata,
            }
        )
        ranqueados = []
        for vid, item in self._datos.get(namespace, {}).items():
            if not _pasa_filtro(item["metadata"], filter):
                continue
            score = _coseno(vector, item["values"])
            meta = _copiar_meta(item["metadata"]) if include_metadata else {}
            ranqueados.append({"id": vid, "score": score, "metadata": meta})
        ranqueados.sort(key=lambda row: row["score"], reverse=True)
        return {"matches": ranqueados[:top_k]}

    def fetch(self, ids, namespace: str, **kwargs):
        validar_namespace(namespace)
        espacio = self._datos.get(namespace, {})
        vectors = {}
        for vid in ids:
            if vid in espacio:
                item = espacio[vid]
                vectors[vid] = {
                    "id": vid,
                    "values": list(item["values"]),
                    "metadata": _copiar_meta(item["metadata"]),
                }
        return {"vectors": vectors, "namespace": namespace}

    def update(self, id, set_metadata, namespace: str, **kwargs):
        validar_namespace(namespace)
        espacio = self._datos.get(namespace, {})
        if id not in espacio:
            raise EsquemaMetadatosError(f"No existe el id {id} en {namespace}.")
        espacio[id]["metadata"].update(set_metadata)
        return SimpleNamespace()

    def delete(self, ids=None, delete_all: bool = False, namespace: str | None = None, **kwargs):
        if not namespace:
            raise EsquemaMetadatosError("delete tiene que indicar namespace=.")
        validar_namespace(namespace)
        if delete_all:
            self._datos[namespace] = {}
            return SimpleNamespace()
        espacio = self._datos.get(namespace, {})
        for vid in ids or []:
            espacio.pop(vid, None)
        return SimpleNamespace()

    def describe_index_stats(self):
        namespaces = {
            ns: SimpleNamespace(vector_count=len(vecs)) for ns, vecs in self._datos.items()
        }
        total = sum(len(vecs) for vecs in self._datos.values())
        return SimpleNamespace(
            dimension=self.dimension,
            total_vector_count=total,
            namespaces=namespaces,
            metric=self.metric,
        )
