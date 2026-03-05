# Research Taste POC

Genera datos de preferencia para decision-making cientifico comparando trayectorias de un modelo **privilegiado** (tiene el paper) vs un modelo **base** (solo tiene datos + pregunta).

## Setup rapido

```bash
# 1. Crear y activar entorno
conda activate research-taste

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Clonar datos de DiscoveryBench (si no esta)
git clone https://github.com/allenai/discoverybench data/discoverybench

# 4. Configurar Azure AI Foundry
cp .env.example .env
# Editar .env con tus credenciales:
#   AZURE_OPENAI_BASE_URL=https://<resource>.openai.azure.com/openai/v1/
#   AZURE_INFERENCE_CREDENTIAL=<api-key>
#   LLM_MODEL=<deployment-name>  (ej: gpt-4.1, DeepSeek-V3.1)
```

## Pipeline

El pipeline tiene 5 pasos. Cada uno es un script independiente.

### 1. Extraer tarea de DiscoveryBench

```bash
python src/extract.py --task biology_fish --output data/tasks/biology_fish.json
```

Tareas disponibles: `biology_fish`, `sociology_bmi`, `economics_immigration`.
Usa `--list` para ver todas las variantes de metadata.

### 2. Generar trayectoria privilegiada (con paper)

```bash
python src/generate_privi.py \
    --task data/tasks/biology_fish.json \
    --paper data/papers/cerezer2023.txt \
    --output trajectories/biology_fish/privi_1.json \
    --steps 6
```

El modelo recibe la pregunta + datos + paper completo. Genera 5-8 pasos de investigacion donde cada decision es justificable sin el paper, pero guiada por el.

### 3. Generar trayectoria base (sin paper)

```bash
python src/generate_base.py \
    --task data/tasks/biology_fish.json \
    --output trajectories/biology_fish/base_1.json \
    --steps 6
```

El modelo recibe solo la pregunta + datos. Sigue un approach generico de data science.

### 4. Extraer pares de divergencia (forks)

```bash
python src/extract_forks.py \
    --privi trajectories/biology_fish/privi_1.json \
    --base trajectories/biology_fish/base_1.json \
    --output trajectories/biology_fish/forks.json
```

Compara ambas trayectorias y encuentra 2-4 puntos donde divergen. Cada fork es un par `chosen` (privi) / `rejected` (base) para DPO.

### 5. Formatear para evaluacion humana

```bash
python src/format_eval.py \
    --forks trajectories/biology_fish/forks.json \
    --output eval/biology_fish_pairs.json
```

Anonimiza y randomiza el orden de los pares para que evaluadores humanos elijan cual refleja mejor juicio cientifico sin saber cual es privi.

### (Opcional) Trayectoria intercalada

```bash
python src/generate_interleaved.py \
    --task data/tasks/biology_fish.json \
    --paper data/papers/cerezer2023.txt \
    --output trajectories/biology_fish/interleaved_1.json \
    --steps 6
```

Base y privi alternan pasos. Genera datos mas diversos forzando al privi a recuperarse de decisiones suboptimas del base.

## Que esperar

Con el `.env` configurado, podemos:

1. **Generar trayectorias** — cada llamada toma ~30-60s y produce un JSON con 5-8 pasos de investigacion
2. **Comparar privi vs base** — el privi deberia elegir mejores metodos (ej: regresion multiple vs random forest) y llegar a conclusiones mas cercanas al gold
3. **Extraer pares DPO** — los forks capturan exactamente donde diverge el juicio cientifico
4. **Validar con humanos** — si evaluadores prefieren consistentemente al privi, la señal es real

## Estructura del proyecto

```
src/
  llm.py          # Cliente Azure AI Foundry v1 API (async, retry, rate limiting)
  common.py       # Utilidades compartidas
  extract.py      # Extrae tareas de DiscoveryBench
  generate_privi.py / generate_base.py / generate_interleaved.py
  extract_forks.py / format_eval.py
prompts/          # System prompts para privi y base
data/tasks/       # Tareas extraidas (JSON)
data/papers/      # Texto de papers fuente
trajectories/     # Trayectorias generadas
eval/             # Pares para evaluacion humana
```

Ver `CLAUDE.md` para convenciones detalladas y `PROJECT.md` para la motivacion completa.
