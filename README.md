# SARAH — Asistente RAG para pacientes con obesidad

Sistema de *Retrieval-Augmented Generation* que responde preguntas de pacientes
sobre obesidad, basándose en un corpus de guías clínicas y literatura científica.
Incluye dos capas de seguridad **antes** del pipeline: detección de crisis de
salud mental y un router *out-of-scope*.

> ⚠️ **Prototipo de investigación.** No reemplaza atención médica profesional.
> Las respuestas de seguridad (crisis) están **pendientes de validación por
> psicólogos** antes de uso con pacientes.

## Arquitectura

```
pregunta → [CRISIS] → [ROUTER out-of-scope] → pipeline RAG → respuesta
```

- **Pipeline RAG:** expansión de query → ruteo por dominio → recuperación híbrida
  (denso bge-m3 + BM25) → fusión RRF → MMR → reranking (bge-reranker-v2-m3) →
  generación con prompt por confianza → guard de fidelidad.
- **Capa de crisis:** regex + similitud semántica con bge-m3. Deriva a líneas de
  ayuda chilenas (*4141, Salud Responde 600 360 7777 opción 2, SAMU 131).
- **Router out-of-scope:** clasificador LLM de 3 vías (SALUD / MIXTA / FUERA) con
  *fallback* por keywords si la API falla.

## Estructura

```
rag_core.py     Lógica completa (pipeline + seguridad + carga de modelos)
app.py          Servicio FastAPI (carga modelos 1 vez, límite de concurrencia)
demo.ipynb      Notebook de desarrollo y demo
requirements.txt
chroma_db/      Índice vectorial (pre-construido)
data/processed/bm25_corpus.json   Corpus BM25
data/pdfs/      PDFs fuente (solo para re-ingestar; NO en el repo)
.env            OPENAI_API_KEY  (NO se sube al repo)
```

## Puesta en marcha

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Crear .env con la clave (NO se versiona)
echo "OPENAI_API_KEY=tu_clave" > .env
```

### Correr el servicio

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Ajusta la concurrencia según la RAM del servidor:

```bash
MAX_CONCURRENCY=3 uvicorn app:app --host 0.0.0.0 --port 8000
```

### Probar

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/preguntar \
     -H "Content-Type: application/json" \
     -d '{"pregunta": "¿Qué es el semaglutide?"}'
```

## Seguridad y credenciales

- **Nunca** subas `.env` (la `OPENAI_API_KEY`). Ya está en `.gitignore`.
- El sistema es **robusto, no infalible**: las capas de seguridad reducen mucho el
  riesgo, pero dependen de un LLM y de reglas que pueden fallar. No debe usarse
  sin supervisión clínica.

## Pendientes conocidos

- Validación del texto de crisis y del umbral semántico con psicólogos.
- El umbral de crisis (0.55) prioriza sensibilidad: puede activarse con
  frustración intensa (no solo crisis real). Decisión clínica a revisar.
- Reproducibilidad: la expansión de query usa temperatura > 0.
- Ablation pendiente del aporte real de RRF.
