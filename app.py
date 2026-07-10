"""
app.py — Servicio FastAPI para SARAH (RAG de obesidad).

Diseño para despliegue en nube (CPU, 24/7, ~20 pacientes de prueba):
  • Los modelos (bge-m3 + reranker) se cargan UNA sola vez al arrancar (lifespan),
    no por petición. Es lo que hace viable la memoria y la latencia.
  • Un semáforo limita cuántas peticiones pesadas corren a la vez (evita OOM
    si llegan varias al mismo tiempo). Ajustable con la env var MAX_CONCURRENCY.
  • El trabajo bloqueante (CPU + llamadas a OpenAI) corre en un threadpool para
    no congelar el event loop.

Correr localmente:
    uvicorn app:app --host 0.0.0.0 --port 8000

Probar:
    curl -X POST http://localhost:8000/preguntar \
         -H "Content-Type: application/json" \
         -d '{"pregunta": "¿Qué es el semaglutide?"}'
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import rag_core as core

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sarah")

# Nº máx de preguntas pesadas simultáneas. 3 es prudente para 20 usuarios en CPU.
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "3"))
_sem = asyncio.Semaphore(MAX_CONCURRENCY)

# Estado compartido: modelos e índice cargados una vez.
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Arrancando SARAH: cargando modelos e índice (una sola vez)...")
    embed_model, reranker, client = core.load_models()
    collection = core.get_collection()
    bm25, bm25_corpus = core.load_bm25_index()
    STATE.update(
        embed_model=embed_model, reranker=reranker, client=client,
        collection=collection, bm25=bm25, bm25_corpus=bm25_corpus,
    )
    log.info(f"SARAH lista. Chunks en índice: {collection.count()}. "
             f"Concurrencia máx: {MAX_CONCURRENCY}")
    yield
    STATE.clear()
    log.info("SARAH detenida.")


app = FastAPI(title="SARAH — Asistente RAG de obesidad", version="1.0", lifespan=lifespan)


class Pregunta(BaseModel):
    pregunta: str = Field(..., min_length=1, max_length=2000)


class Respuesta(BaseModel):
    respuesta: str
    contexto: str


@app.get("/health")
async def health():
    """Chequeo de salud del servicio (para monitoreo / load balancer)."""
    if not STATE:
        raise HTTPException(503, "modelos aún no cargados")
    return {"status": "ok", "chunks": STATE["collection"].count()}


@app.post("/preguntar", response_model=Respuesta)
async def preguntar(body: Pregunta):
    """
    Recibe la pregunta de un paciente y devuelve la respuesta del sistema.
    Pasa por: CRISIS → ROUTER out-of-scope → pipeline RAG (safe_answer_query).
    """
    q = body.pregunta.strip()
    if not q:
        raise HTTPException(400, "La pregunta está vacía.")

    async with _sem:  # respeta el límite de concurrencia
        try:
            answer, context = await run_in_threadpool(
                core.safe_answer_query,
                q,
                STATE["collection"], STATE["bm25"], STATE["bm25_corpus"],
                STATE["embed_model"], STATE["reranker"], STATE["client"],
            )
        except Exception as e:
            log.exception("Error procesando la pregunta")
            raise HTTPException(500, "Ocurrió un error procesando tu pregunta.") from e

    return Respuesta(respuesta=answer, contexto=context)
