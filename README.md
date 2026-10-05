# RAG híbrido en Pinecone

Servicio en Python que ingesta las políticas internas de TechCorp en un índice Pinecone Serverless, recupera con un `EnsembleRetriever` (BM25 + vectores) y mide Precision@5 y Recall@5 sobre 5 preguntas.

El embedding es OpenAI `text-embedding-3-small` (1536 dimensiones, cosine). Sin API key el mismo flujo corre en memoria, con un embedding determinista del mismo ancho. Con `--live` crea el índice Serverless en esa dimensión y embeddea con OpenAI.

## Quick path

1. Instalar. Un `pip install -r requirements.txt` alcanza (incluye `pytest`, `pytest-asyncio` y `pytest-mock`).

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

**Linux/macOS (bash/zsh):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

2. Chequeo offline (no llama a Pinecone ni baja el modelo) y tests:

```
python validacion.py
python -m pytest -v
python evaluate.py
python main.py
```

`evaluate.py` imprime Recall@5 promedio 100% y Precision@5 promedio 20%. En las cinco preguntas el archivo esperado es el primero de los 5 recuperados.

## Replicar el índice en Pinecone

`.env` (no se versiona) necesita:

```
PINECONE_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
INDEX_NAME=techcorp-rag-hibrido
```

Para embeddear hace falta `OPENAI_API_KEY` (`text-embedding-3-small`, índice de 1536). `ANTHROPIC_API_KEY` es la alternativa de proveedor que pide la letra; no cambia el ancho del índice. Ninguna clave se imprime.

El ejemplo está en `.env.example`.

**Windows (PowerShell):**

```powershell
python init_index.py --live --dimension 1536
python main.py --live --chunk-size 600 --chunk-overlap 100 --batch-size 100 --k 5
python evaluate.py --live --k 5
```

**Linux/macOS (bash/zsh):**

```bash
python init_index.py --live --dimension 1536
python main.py --live --chunk-size 600 --chunk-overlap 100 --batch-size 100 --k 5
python evaluate.py --live --k 5
```

`init_index.py --live` hace esto:

1. Lee `PINECONE_API_KEY`, `OPENAI_API_KEY` o `ANTHROPIC_API_KEY`, e `INDEX_NAME` (default `techcorp-rag-hibrido`).
2. Arma `indices_existentes = [i["name"] for i in pc.list_indexes()]`.
3. Si el nombre no está, crea un índice Serverless con `ServerlessSpec(cloud="aws", region="us-east-1")`, `metric="cosine"` y `dimension=1536` (`text-embedding-3-small`). Espera a que quede ready.
4. Si ya existe, compara dimensión y métrica. Un índice de 512 o 768, o con métrica distinta, se aborta. No se reintenta y no se borra: hay que elegir otro `INDEX_NAME` o eliminarlo en la consola de Pinecone.

`main.py --live` carga `.txt`, `.md`, `.json` y `.pdf` de `data/` (`DirectoryLoader` + `TextLoader` para texto y Markdown, y `PyPDFLoader` para PDF), los parte con `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (default `chunk_size=600`, `chunk_overlap=100`) y los sube con `PineconeVectorStore.from_documents` al namespace `politicas-internas`. El texto queda en `metadata["text"]`, junto con `fuente`, `pagina` y `etiquetas`.

El mismo par embedding/índice tiene que usarse al indexar y al consultar. `--live` en los dos comandos usa `text-embedding-3-small` (1536). El modo offline no toca Pinecone ni OpenAI.

`--chunk-size`, `--chunk-overlap`, `--batch-size`, `--k` y `--dimension` llegan al splitter, al `upsert` y al retriever. El default no los pisa.

## Archivos del repositorio

| Artefacto | Dónde está |
|-----------|------------|
| `asegurar_indice` (Serverless, lista de índices, mismatch) | `init_index.py` |
| `DirectoryLoader` / `TextLoader`, chunks, `from_documents`, CRUD | `ingesta.py` |
| `class RAGSystem` y `obtener_top_k` | `rag_system.py` |
| `BM25Retriever` + `PineconeVectorStore` en un `EnsembleRetriever` | `rag_system.py` |
| `evaluar` (Recall@5, Precision@5 y los promedios) | `evaluate.py` |
| Golden set `{"pregunta", "documento_id_esperado"}` | `evaluate.py` y `data/golden_set.json` |
| Políticas de TechCorp (`.txt`, `.md`, `.json`, `.pdf`) | `data/` |
| Errores 401 / 429 / red / mismatch / schema / truncado | `errors.py`, `reintentos.py` |
| `OpenAIEmbeddings` `text-embedding-3-small` (1536) o el sustituto offline | `embeddings.py` |
| Demo | `main.py` |
| Chequeo offline | `validacion.py` |
| Tests | `tests/` |
| `pytest.ini` | `testpaths = tests` · `asyncio_mode = auto` · `addopts = -ra -q` |

## Cómo se cubre cada criterio

### Configuración e infraestructura cloud (Pinecone)

`init_index.py` lee `PINECONE_API_KEY`, `OPENAI_API_KEY` o `ANTHROPIC_API_KEY`, e `INDEX_NAME` desde el `.env`. El archivo real no se versiona; el ejemplo vacío es `.env.example`. Si falta la clave de Pinecone, el proceso sale con `401/key: falta PINECONE_API_KEY en las variables de entorno.` y no imprime el valor.

Si el índice no existe, lo crea Serverless (`ServerlessSpec`, aws, `us-east-1`, cosine, dimensión 1536, el ancho de `text-embedding-3-small`). Si ya existe con otro ancho (512, 768) o con otra métrica, aborta y no reintenta.

Evidencia: `test_leer_config_toma_el_entorno` · `test_crea_indice_serverless_con_la_dimension_del_modelo` · `test_mismatch_de_dimension_no_reintenta` · `test_falta_api_key`

### Pipeline de ingesta y gestión de metadatos

`ingesta.py` carga documentos técnicos en `.txt`, `.md`, `.json` y `.pdf`. Los parte con `RecursiveCharacterTextSplitter.from_tiktoken_encoder`. El default es 600 tokens y 100 de solapamiento (dentro de 500–800) y el valor de la llamada es el que llega al splitter.

Cada fragmento persiste el texto original en `metadata["text"]`, más `fuente`, `pagina` y `etiquetas` de categoría. La subida usa `PineconeVectorStore.from_documents` en el namespace `politicas-internas`, en lotes (`batch_size`, default 100). 429 y cortes de red se reintentan; 401 y un vector de otro ancho, no.

Evidencia: `test_carga_txt_markdown_json_y_pdf` · `test_splitter_usa_el_chunk_size_de_la_llamada` · `test_metadata_sale_del_nombre_de_archivo` · `test_subida_guarda_texto_y_filtra_por_namespace` · `test_batch_size_llega_al_upsert_y_no_se_clava_en_100`

### Implementación del recuperador híbrido

`RAGSystem` encapsula un `EnsembleRetriever`. Adentro hay un `BM25Retriever` (palabras literales: `2FA`, nombres propios) y el retriever de `PineconeVectorStore` (similitud de vectores). Los pesos son `[0.5, 0.5]`. `obtener_top_k` recibe la consulta y devuelve 5 documentos con `contenido`, `fuente` y `categoria`. El `k` de la llamada llega al BM25 y a `search_kwargs`. La búsqueda lleva `namespace`.

Evidencia: `test_k_de_la_llamada_llega_al_bm25_y_a_pinecone` · `test_token_raro_queda_primero_y_el_default_es_top_5` · `test_namespace_no_mezcla_al_tenant`

### Evaluación cuantitativa de métricas

`evaluate.py` lee 5 pares `{"pregunta", "documento_id_esperado"}` de `data/golden_set.json`. Cada consulta pide el top-5. Recall@5 vale 1 si el archivo esperado está entre esos 5. Precision@5 es la proporción de esos 5 que pertenecen a ese archivo. El resumen se imprime en consola: Recall@5 promedio 1 y Precision@5 promedio 0.20.

Evidencia: `test_golden_set_tiene_cinco_preguntas` · `test_corpus_real_recupera_el_documento_esperado` · `evidencias/03-evaluate.txt`

## Contrato de namespaces y metadatos

- **Corpus:** `politicas-internas`. Un solo espacio para las políticas de TechCorp.
- **Entornos y tenants, aparte:** `ns-dev`, `ns-staging`, `ns-prod` y `ns-cliente-<id>`. Un upsert no mezcla dos espacios. Query, fetch, update y delete llevan `namespace=`.
- **Metadata:** `text` (el fragmento), `source` (por ejemplo `politica_vacaciones.txt`), `categoria` (por ejemplo `politica vacaciones`), `chunk_id` (entero).
- **Filtro:** `{"source": {"$eq": "politica_vacaciones.txt"}}` junto con `include_metadata=True`.
- **Métrica:** cosine. `text-embedding-3-small` es de 1536. Un índice de 512 o 768 no sirve para este embedding.

## Por qué 1536, cosine y búsqueda híbrida

`text-embedding-3-small` se compara por coseno y devuelve 1536 números. Ese ancho es el que se pasa a `create_index` y a `OpenAIEmbeddings(dimensions=1536)`. El corte default de 600 tokens (piso 500, techo 800) y el overlap de 100 llegan tal cual a `from_tiktoken_encoder`. BM25 sostiene términos literales (`2FA`, `Soporte Técnico Nivel 1`); el vector sostiene el parafraseo. El `EnsembleRetriever` los junta con pesos iguales.

## Evaluación

Cinco preguntas, cada una con un solo `documento_id_esperado` (el nombre del `.txt`).

- **Recall@5** = 1 si esa fuente aparece entre lo recuperado, si no 0. Con un único documento relevante no hay un valor intermedio.
- **Precision@5** = coincidencias / cantidad recuperada.

El corpus tiene 9 fragmentos y `k=5`, así que cada consulta devuelve 5. El archivo esperado entra una vez: Precision@5 = 1/5 = 0.20. En las cinco preguntas ese archivo es el primero: Recall@5 = 1.

Salida de `python evaluate.py`:

```
RECALL@5 PROMEDIO:    100.0%
PRECISION@5 PROMEDIO: 25.0%
Precision@5: 0.2000
Recall@5: 1.0000
```

`obtener_top_k` devuelve una lista de dicts con `contenido`, `fuente` y `categoria`.

## Manejo de errores personalizados

`main.py`, `evaluate.py` e `init_index.py` capturan `RAGCloudError` e imprimen `Error controlado: ...` con código de salida 1.

### 401 / key

**Mensaje:** `401/key: falta PINECONE_API_KEY en las variables de entorno.`

**Cómo reproducirlo:** `python init_index.py --live` sin `PINECONE_API_KEY`.

**Test:** `tests/test_init_index.py::test_falta_api_key`

Si la API responde unauthorized: `401/key: clave de Pinecone inválida o ausente. Detalle: Unauthorized: invalid API key`. No se reintenta (`test_401_en_list_indexes_no_reintenta`, `test_401_en_upsert_no_reintenta`).

### 429 / cuota

**Mensaje:** `429/cuota: Pinecone rechazó la operación por cuota o rate limit. Detalle: rate limit / quota exceeded`

**Cómo reproducirlo:** un `list_indexes` o un `upsert` con status 429. Hay hasta 3 intentos; entre el primero y el segundo espera 0.5s y entre el segundo y el tercero 1s (en tests la espera es 0). Si el segundo responde bien, la operación sigue.

**Test:** `tests/test_init_index.py::test_429_agotado` · `test_429_reintenta_y_luego_crea` · `tests/test_ingesta.py::test_429_en_upsert_reintenta_y_recupera`

### red / timeout

**Mensaje:** `red/timeout: no se pudo hablar con Pinecone. Detalle: Request timed out talking to Pinecone`

**Cómo reproducirlo:** un `TimeoutError` en el upsert. Se reintenta igual que el 429.

**Test:** `tests/test_ingesta.py::test_timeout_en_upsert_se_agota`

Si el índice no pasa a ready: `red/timeout: el índice techcorp-rag-hibrido no quedó ready en 0s.` (`test_indice_no_ready_es_timeout`).

### Mismatch de dimensiones

**Mensaje:** `Mismatch de dimensiones: el índice techcorp-rag-hibrido es 768D y se pidió 1536D.`

**Cómo reproducirlo:** el índice ya existe con el ancho de otro modelo. No se reintenta y no se llama a `create_index`.

**Test:** `tests/test_init_index.py::test_mismatch_de_dimension_no_reintenta`

`dimension=0`: `Mismatch de dimensiones: dimension=0 es inválida (tiene que coincidir con el embedding, p.ej. 1536).` (`test_dimension_invalida`).

Vector de otro largo: `Mismatch de dimensiones: el vector es 2D y el índice es 8D.` (`test_mismatch_de_vector_no_reintenta`).

Pedir otro ancho con este modelo: `Mismatch de dimensiones: text-embedding-3-small usa 1536D y se pidió 384D.` (`test_openai_rechaza_otro_ancho`). Sin `OPENAI_API_KEY`: `401/key: falta OPENAI_API_KEY en las variables de entorno.` (`test_openai_sin_clave`).

### Mismatch de métrica

**Mensaje:** `Mismatch de métrica: se pidió euclidean y text-embedding-3-small usa cosine.`

**Cómo reproducirlo:** pedir `metric="euclidean"`, o apuntar a un índice que ya es `euclidean`.

**Test:** `tests/test_init_index.py::test_metrica_euclidean_no_llega_a_crear` · `test_mismatch_de_metrica_en_indice_existente`

### Schema drift

**Mensaje:** `Schema drift: ['date_created'] no pertenece al contrato de metadatos.`

**Cómo reproducirlo:** un documento cuya metadata trae `date_created`, `ingest_date`, `created_at` o `fecha`.

**Test:** `tests/test_ingesta.py::test_schema_drift_en_el_documento` · `test_pydantic_rechaza_campo_extra`

### Salida truncada

**Mensaje:** `Salida truncada: el chunk 0 supera 20000 caracteres y no se recorta en silencio.`

**Cómo reproducirlo:** un fragmento de más de 20000 caracteres. No se recorta para entrar en la metadata. El test baja el tope a 10 caracteres (`test_salida_truncada_no_recorta`).

### Consulta vacía

**Mensaje:** `Consulta vacía: no hay texto para recuperar.`

**Cómo reproducirlo:** `rag_system.obtener_top_k("   ")`.

**Test:** `tests/test_rag.py::test_consulta_vacia`

### Lote, k o chunk fuera de rango

**Mensaje:** `chunk_size debe ser un entero >= 500 y <= 800. Recibido: 499.`

**Cómo reproducirlo:** `construir_splitter(chunk_size=499)` o `chunk_size=801`. `batch_size=1001`: `batch_size debe ser un entero entre 1 y 1000 (límite de la API). Recibido: 1001.` `k=0`: `k debe ser un entero entre 1 y 100. Recibido: 0.`

**Test:** `tests/test_ingesta.py::test_chunk_size_fuera_de_rango` · `test_batch_size_y_k_fuera_de_limite`

### Error no transitorio

**Mensaje:** `Error de Pinecone: boom`

**Cómo reproducirlo:** una excepción que no es 401, 429 ni red. No se reintenta.

**Test:** `tests/test_ingesta.py::test_error_no_transitorio_no_reintenta`

## Evidencias

| Archivo | Qué muestra |
|---------|-------------|
| `evidencias/01-validacion-offline.txt` | `python validacion.py`: cadenas, familias de error y Precision@5 / Recall@5. |
| `evidencias/02-pytest.txt` | `python -m pytest -o addopts= -v -ra`. |
| `evidencias/03-evaluate.txt` | `python evaluate.py`. |
| `evidencias/04-main.txt` | `python main.py`: ingesta, una consulta y el reporte. |
| `evidencias/05-init-index.txt` | `python init_index.py`: índice local, sin crear nada en Pinecone. |

## Checklist

- [x] `.env.example` con `PINECONE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` e `INDEX_NAME=techcorp-rag-hibrido` (el `.env` real no se versiona)
- [x] `init_index.py` crea el índice Serverless de 1536 / cosine (`text-embedding-3-small`) si no existe y aborta si no coincide
- [x] `DirectoryLoader` + `TextLoader` sobre los `.txt` de TechCorp
- [x] `from_tiktoken_encoder` con `chunk_size` y `chunk_overlap` de la llamada (default 600 / 100)
- [x] Carga de `.txt`, `.md`, `.json` y `.pdf`
- [x] Texto original en `metadata["text"]`, más `fuente`, `pagina`, `etiquetas`, `source` y `chunk_id`
- [x] Cada consulta de evaluación devuelve 5 fragmentos (Precision@5 sobre esos 5)
- [x] Namespace `politicas-internas`, y también `ns-dev` / `ns-staging` / `ns-prod` / `ns-cliente-<id>`
- [x] `PineconeVectorStore.from_documents` y lotes con reintentos ante 429 y red
- [x] CRUD: fetch, update de metadatos, delete por id y `delete_all` por namespace
- [x] `class RAGSystem` con `obtener_top_k`, `BM25Retriever` y `EnsembleRetriever`
- [x] `evaluar` con Recall@5 y Precision@5 sobre 5 preguntas
- [x] Resumen de métricas en consola y en este README
- [x] pytest + mocks, sin API real
- [x] Una subsección por familia de error, con el mensaje, cómo reproducirlo y el test
- [x] `evidencias/` con la salida real de los comandos
