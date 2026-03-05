# CHANGELOG — Research Taste POC

Registro cronologico de cambios significativos. Formato: fecha, descripcion, archivos afectados.

---

## 2026-03-05 — Setup inicial del proyecto

### Estructura y scaffolding
- Creada estructura completa de directorios: `src/`, `prompts/`, `data/`, `trajectories/`, `eval/`, `results/`
- Creados `.gitkeep` para preservar carpetas vacias en git
- Creado `.env.example` con variables para Azure OpenAI
- Creado `requirements.txt`: openai, pandas, python-dotenv
- Actualizado `.gitignore` con entradas del proyecto (data/discoverybench, trajectories JSON, etc.)

### Prompts
- `prompts/privi_system.txt` — System prompt para modelo privilegiado (tiene paper)
- `prompts/base_system.txt` — System prompt para modelo base (sin paper)
- `prompts/step_template.txt` — Template para generacion paso a paso

### Scripts del pipeline
- `src/llm.py` — Cliente Azure OpenAI compartido (async, semaphore rate limiting, retry con backoff, timeout, JSON extraction)
- `src/common.py` — Utilidades compartidas (load_task, load_paper, build_dataset_summary, save_json, format_steps)
- `src/extract.py` — Extractor de tareas desde metadata de DiscoveryBench
- `src/generate_privi.py` — Generador de trayectorias privilegiadas (con paper)
- `src/generate_base.py` — Generador de trayectorias base (sin paper)
- `src/generate_interleaved.py` — Generador de trayectorias intercaladas (base/privi alternan pasos)
- `src/extract_forks.py` — Extractor de divergencias entre privi y base via LLM
- `src/format_eval.py` — Formateador de pares para evaluacion humana ciega (randomiza orden)

### Infraestructura
- Clonado DiscoveryBench en `data/discoverybench/` (estructura real: `real/train/<folder>`)
- Creado conda env `research-taste` (Python 3.11.14)
- Instaladas dependencias: openai 2.24.0, pandas 3.0.1, python-dotenv 1.2.2

### Migracion anthropic -> Azure OpenAI
- Todos los scripts migrados de `anthropic` SDK a `openai` SDK (AsyncAzureOpenAI)
- Centralizado cliente LLM en `src/llm.py` con patron async + retry + rate limiting
- Extraida logica compartida a `src/common.py` para eliminar duplicacion

### Documentacion
- `CLAUDE.md` — Actualizado con tech stack Azure OpenAI, estructura con llm.py y common.py
- `PROJECT.md` — Descripcion completa del proyecto y motivacion (sin cambios)
- `TODO.md` — Tracking de tareas (creado)
- `CHANGELOG.md` — Este archivo (creado)

### Hallazgos
- DiscoveryBench estructura real: `discoverybench/real/train/<folder_name>/` (no `real/<domain>/`)
- Carpetas: `evolution_freshwater_fish`, `nls_bmi`, `nls_bmi_raw`, `immigration_offshoring_effect_on_employment`
- Cada carpeta tiene `metadata_N.json` + archivos de datos (CSV o DTA)
- `extract.py` necesita adaptarse a esta estructura (pendiente)
