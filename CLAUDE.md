# CLAUDE.md — Research Taste POC

## Document maintenance system

Este proyecto usa 4 documentos vivos que DEBEN mantenerse actualizados en cada sesion de trabajo:

| Documento | Proposito | Cuando actualizar |
|-----------|-----------|-------------------|
| `CLAUDE.md` | Instrucciones para Claude, arquitectura, convenciones | Cuando cambia el tech stack, estructura, o convenciones |
| `PROJECT.md` | Descripcion completa del proyecto, motivacion, approach | Cuando cambia el scope o la estrategia del proyecto |
| `TODO.md` | Tareas pendientes, en progreso, completadas | **Cada vez que se completa, agrega o bloquea una tarea** |
| `CHANGELOG.md` | Registro cronologico de cambios | **Cada vez que se hace un cambio significativo** |

### Reglas obligatorias

1. **Al completar cualquier tarea**: mover de "Pendiente" / "En progreso" a "Completado" en TODO.md
2. **Al crear algo nuevo**: agregar entrada en CHANGELOG.md con fecha, descripcion y archivos afectados
3. **Al descubrir algo**: agregar hallazgos relevantes en CHANGELOG.md bajo la fecha actual
4. **Al encontrar un blocker**: agregar en TODO.md bajo "Bloqueado" con la razon
5. **Antes de hacer commit**: verificar que TODO.md y CHANGELOG.md estan al dia

## Project overview

This project generates preference training data for scientific decision-making by comparing trajectories from a "privileged" model (has the paper) vs a "base" model (only has data + question). We use DiscoveryBench tasks as our source.

**We are in POC phase: generating and validating data, NOT training models yet.**

## Tech stack

- Python 3.11+ (conda env: `research-taste`)
- openai SDK via Azure OpenAI (AsyncAzureOpenAI for all LLM calls)
- pandas (for reading DiscoveryBench CSVs)
- python-dotenv (for .env config)
- json (trajectory storage)
- No ML frameworks yet — this is data generation only

## Key concepts

- **privi**: the model instance that receives the full paper + data + question. It knows the answer.
- **base**: the model instance that receives only data + question. It doesn't know the answer.
- **trajectory**: a sequence of 5-8 research steps, each with reasoning, action, and outcome.
- **fork**: a point where privi and base diverge from the same state. Creates a DPO preference pair.
- **interleaved trajectory**: privi and base alternate steps. Base introduces "noise", privi recovers.

## Project structure

```
research-taste/
├── CLAUDE.md              # This file — instructions for Claude
├── PROJECT.md             # Full project description and motivation
├── TODO.md                # Task tracking (pendiente / en progreso / completado)
├── CHANGELOG.md           # Chronological record of changes
├── .env.example           # Azure OpenAI config template
├── requirements.txt       # Python dependencies
├── data/
│   ├── discoverybench/    # Clone of allenai/discoverybench
│   ├── papers/            # Source paper text files
│   └── tasks/             # Extracted task JSONs
├── src/
│   ├── llm.py             # Shared Azure OpenAI client (async, retry, rate limit)
│   ├── common.py          # Shared utilities (load task, build summaries, etc.)
│   ├── extract.py         # Parse DiscoveryBench metadata into task objects
│   ├── generate_privi.py  # Generate privileged trajectories via API
│   ├── generate_base.py   # Generate base trajectories via API
│   ├── generate_interleaved.py  # Alternating privi/base steps
│   ├── extract_forks.py   # Find divergence points, create DPO pairs
│   └── format_eval.py     # Blind and randomize pairs for human eval
├── prompts/
│   ├── privi_system.txt   # System prompt for privileged model
│   ├── base_system.txt    # System prompt for base model
│   └── step_template.txt  # Template for each trajectory step
├── trajectories/          # Generated trajectories (JSON)
├── eval/                  # Blinded pairs for human evaluation
└── results/               # Human ratings
```

## How to run

### Setup
```bash
# Activate conda env
conda activate research-taste

# Clone DiscoveryBench data (if not already done)
git clone https://github.com/allenai/discoverybench data/discoverybench

# Install dependencies
pip install -r requirements.txt

# Configure Azure OpenAI (copy .env.example to .env and fill in values)
cp .env.example .env
```

### Generate trajectories for a task
```bash
# Extract task info from DiscoveryBench
python src/extract.py --task biology_fish --output data/tasks/biology_fish.json

# Generate privileged trajectory (with paper context)
python src/generate_privi.py --task data/tasks/biology_fish.json --paper data/papers/cerezer2023.txt --output trajectories/biology_fish/privi_1.json

# Generate base trajectory (without paper)
python src/generate_base.py --task data/tasks/biology_fish.json --output trajectories/biology_fish/base_1.json

# Extract fork pairs
python src/extract_forks.py --privi trajectories/biology_fish/privi_1.json --base trajectories/biology_fish/base_1.json --output trajectories/biology_fish/forks.json

# Format for human eval (blinded)
python src/format_eval.py --forks trajectories/biology_fish/forks.json --output eval/biology_fish_pairs.json
```

## DiscoveryBench data structure

Data lives in `data/discoverybench/discoverybench/real/train/<folder_name>/`. Each folder has:
- `metadata_N.json` files (one per query/task variant)
- Data files (CSV or DTA)

Folder mapping for our 3 POC tasks:
- `evolution_freshwater_fish/` — biology fish task
- `nls_bmi/` — sociology BMI task
- `immigration_offshoring_effect_on_employment/` — economics task

Each metadata JSON has:
- `datasets`: array with dataset name, description, column definitions
- `queries`: array with `question` and `true_hypothesis`
- `workflow_tags`, `domain_knowledge`

## Three POC tasks

### 1. Biology — Fish speciation (Cerezer et al., 2023, Nature Communications)
- **Dataset**: `body-size-evolution-in-south-american-freshwater-fishes.csv` (460 rows, 20 cols)
- **Question**: What drives spatial variation in speciation rates?
- **Gold answer**: MBL_evol (max body length evolution rate) is the strongest predictor (coef 0.82). Climate and area are NOT significant.
- **Why interesting**: The "obvious" analysis (RF feature importance) gives a misleading answer. The correct approach (multiple regression with hierarchical partitioning) is non-default.

### 2. Sociology — BMI and time preference (NLSY79 data)
- **Dataset**: `nls_bmi_processed.csv` (~12K rows, 9 cols)
- **Question**: What factors related to time preference are associated with higher BMI?
- **Gold answer**: DISSAVED (coef 0.36) and SAMESAVE (coef 0.49) are significant predictors of BMI, controlling for age, income, gender, race.
- **Why interesting**: Requires understanding that savings behavior proxies time preference (conceptual leap). Bivariable correlations are weak; effect only appears with proper controls.

### 3. Economics — Immigration and offshoring (multi-dataset)
- **Datasets**: `offshoring_iv_mar2.dta` + `immi_popimputed_00_07.dta` (need integration)
- **Question**: How does immigration ease affect offshoring employment share?
- **Gold answer**: Per unit increased ease of immigration reduces 0.1059 unit of offshoring employment share. But immigration has NO significant effect on native employment.
- **Why interesting**: Requires merging datasets, instrumental variable logic, and interpreting null results as meaningful.

## Trajectory generation guidelines

When generating trajectories, each step must have:

1. **action_type**: one of explore, analyze, interpret, decide, verify
2. **reasoning**: WHY this step makes sense given what we know so far (2-4 sentences)
3. **action**: WHAT specifically to do (concrete analysis, not vague)
4. **expected_outcome**: what the researcher expects to find BEFORE looking
5. **actual_outcome**: what they actually find (can diverge from expectation)

The privi's trajectory should:
- Make decisions that are JUSTIFIABLE without the paper (not "I know the answer is X so let's check X")
- Use the paper to SELECT among reasonable options, not to invent options that only make sense post-hoc
- Still follow a natural research progression (explore -> analyze -> interpret -> verify)

The base's trajectory should:
- Represent what a competent but generic data scientist would do
- Use common defaults (random forest, correlation matrix, standard feature importance)
- NOT be intentionally bad — it should be reasonable but miss the non-obvious insights

## Coding style

- Python, simple scripts, no frameworks
- All LLM calls go through `src/llm.py` (Azure OpenAI, async with retry/rate limiting)
- Shared utilities in `src/common.py`
- JSON for all data storage
- Keep it minimal — this is a POC, not production code
- Print progress to stdout, don't over-engineer logging
- Each script should be runnable independently from `src/` or project root

## Azure OpenAI conventions

- Client: `AsyncAzureOpenAI` (always async)
- Auth: API key via `AZURE_API_KEY` env var
- Endpoint: `AZURE_API_BASE` (sin trailing slash)
- `model` param = deployment name in Azure, NOT model name
- Use `max_completion_tokens` (not `max_tokens`)
- Temperature: omit or set to 1.0
- Rate limiting: `asyncio.Semaphore` in `src/llm.py`
- Retry: exponential backoff (2^attempt seconds)

## Important: what NOT to do

- Don't build a training pipeline yet
- Don't optimize prompts prematurely — get the basic trajectories first
- Don't try to automate human evaluation — we need real humans looking at the pairs
- Don't use DiscoveryBench's synthetic tasks — only use DB-REAL (real papers)
