# Research Taste via Privileged Self-Distillation

## What is this project?

We want to teach LLMs to make better scientific decisions — not just answer science questions, but to take the right *steps* during an investigation. Which analysis to run, what to control for, when to change approach, how to interpret ambiguous results.

The core idea: take the **same model**, give one copy access to a published paper's full results (the "privi"), give the other copy only the raw data and a research question (the "base"). Both generate step-by-step research trajectories. The privi makes better decisions because it knows how the story ends. We extract preference pairs from the divergences and train with DPO.

## Why this matters

LLMs today fail at open-ended scientific tasks not because they lack knowledge, but because they lack **judgment about what's relevant**. They go off on tangents, pick default methods instead of appropriate ones, don't control for confounders, and declare success on weak evidence. This is the "research taste" gap.

Current approaches to this problem:
- **ResearchPlanGen** (Meta, Dec 2025): trains models to generate research plans, but plans are static and don't involve actual data
- **DiscoveryBench** (Allen AI, 2024): benchmarks LLMs on data-driven discovery tasks, but only evaluates final answers, not the process
- **ResearchGym** (2025): tests agents on full research loops, but only for ML/code tasks

Nobody is generating **trajectory-level preference data for scientific decision-making**. That's the gap.

## The approach: Privileged Self-Distillation

### What the privi gets (privileged information)
- The research question
- The raw dataset(s)
- The **full published paper**: methods, results, interpretation, what worked, what didn't

### What the base gets
- The research question
- The raw dataset(s)
- Optional: domain knowledge hints (e.g., "saving behavior proxies time preference")

### The pipeline

**Step 1: Generate privi trajectory.**
The privi generates a 5-8 step research trajectory, knowing how the paper turned out. Each step is: what analysis to do, why, what you expect to find.

**Step 2: Generate base trajectory.**
The base generates its own trajectory from the same starting point. It doesn't know the paper's results.

**Step 3: Fork-based DPO pairs.**
At each step where the privi and base diverge, we extract a preference pair:
- **chosen**: the privi's decision at that step
- **rejected**: the base's decision at that step
Both from the same state (everything up to that step is identical).

**Step 4 (optional): Interleaved trajectories for diversity.**
The base generates a step, then the privi continues from that (possibly wrong) state. This forces the privi to explore paths it would never take on its own, creating more diverse training data.

### Why this is NOT just SFT with expert trajectories

Three reasons:
1. **No covariate shift**: DPO pairs come from states the base actually visits (especially with interleaving), not idealized expert states
2. **Negative signal**: the base's bad decisions are explicitly contrasted with good ones — SFT only shows what TO do, DPO also shows what NOT to do
3. **The base as perturbator**: in interleaved mode, the base's "errors" force the privi to explore diverse recovery paths, preventing data collapse

### Relevant prior work
- **OEC / DAgger** (Lauffer et al., Dec 2024): student starts, expert finishes — 14% improvement over SFT in SWE tasks
- **LUPI** (Vapnik): learning using privileged information, theoretical foundation for teacher-with-extra-context
- **HER** (Andrychowicz et al., 2017): hindsight experience replay — learning from failures by relabeling goals
- **ReSyn** (2026): generator-verifier gap — verifying is easier than generating, so privileged verification gives better signal

## Data source: DiscoveryBench (Allen AI)

We use DiscoveryBench DB-REAL as our paper source. It provides:
- **264 tasks** across 6 domains (sociology, biology, economics, humanities, engineering, meta-science)
- **Real public datasets** (CSV files downloadable from the repo)
- **Gold hypotheses** (the correct answer the paper found)
- **Workflow tags** (what type of analysis: regression, feature engineering, data cleaning, etc.)
- **Domain knowledge hints** (optional contextual info)

Repo: https://github.com/allenai/discoverybench
HuggingFace: https://huggingface.co/datasets/allenai/discoverybench

### Example task structure (biology domain)

**Dataset**: `body-size-evolution-in-south-american-freshwater-fishes.csv`
- 460 rows (sub-basins), 20 columns
- Variable dependiente: BAMM_speciation (speciation rate)
- Predictors: 5 morphological evolution rates, 4 climate variables, 4 habitat variables, species diversity

**Paper**: Cerezer et al. (2023), Nature Communications 14, 5515
"Accelerated body size evolution in upland environments is correlated with recent speciation in South American freshwater fishes"

**Question**: "Is the maximum body length evolution the most impactful factor in explaining the speciation rates?"

**Gold hypothesis**: "The rate of maximum body length evolution emerged as the most influential factor explaining spatial variation in speciation rates. The relationship is positive with linear coefficient 0.82."

**Why it's non-obvious**: The dominant hypothesis in the field was that climate (temperature) and area drive speciation. This paper shows those factors are negligible once you control for morphological evolution. A model without the paper would likely conclude that temperature and area matter (they correlate with speciation, but the effect is spurious).

## POC scope: 3 tasks

Start with 3 tasks from DiscoveryBench that have:
1. Non-obvious findings (privi/base should diverge meaningfully)
2. Interesting analytical decisions (not just "run a regression")
3. Different domains (to check generalization of the pattern)

**Candidate tasks:**
1. **Biology — fish speciation** (Cerezer et al., 2023): multiple regression with many predictors, counterintuitive result that climate doesn't matter
2. **Sociology — BMI and time preference** (NLSY79 data): requires understanding that savings behavior proxies time preference, needs subgroup analysis
3. **Economics — immigration and offshoring** (multiple datasets): requires integrating two datasets, instrumental variables, null results are meaningful

## What we're measuring in the POC

**Primary question**: Do the privi/base fork pairs capture meaningful differences in scientific judgment?

**How we measure it**:
1. Generate 5-8 step trajectories from privi and base for each of the 3 tasks
2. Extract fork pairs at each divergence point
3. Present pairs (blinded) to 3-5 researchers: "Which continuation reflects better scientific judgment?"
4. If humans consistently prefer the privi's choice → the signal is real
5. If humans can't tell → the approach doesn't work

**We are NOT training anything in the POC.** We're validating that the data generation mechanism produces useful preference pairs before investing in training infrastructure.

## Project structure

```
research-taste-poc/
├── CLAUDE.md                 # Instructions for Claude Code
├── PROJECT.md                # This document
├── data/
│   ├── discoverybench/       # Cloned from allenai/discoverybench
│   └── papers/               # Full text of source papers (PDFs)
├── src/
│   ├── extract.py            # Extract task info from DiscoveryBench metadata
│   ├── generate_privi.py     # Generate privileged trajectories
│   ├── generate_base.py      # Generate base trajectories
│   ├── generate_interleaved.py  # Generate interleaved trajectories
│   ├── extract_forks.py      # Extract fork pairs from trajectory pairs
│   └── format_eval.py        # Format pairs for human evaluation (blinded)
├── trajectories/
│   ├── task_biology_fish/
│   │   ├── privi_traj_1.json
│   │   ├── base_traj_1.json
│   │   ├── interleaved_traj_1.json
│   │   └── forks.json
│   ├── task_sociology_bmi/
│   └── task_economics_immigration/
├── eval/
│   └── human_eval_pairs.json # Blinded pairs for human evaluation
└── results/
    └── human_ratings.json    # Collected human judgments
```

## Trajectory format

Each trajectory is a JSON with this structure:

```json
{
  "task_id": "biology_fish_speciation",
  "model": "claude-sonnet-4-5-20250929",
  "mode": "privi",  // or "base" or "interleaved"
  "paper_in_context": true,  // false for base
  "steps": [
    {
      "step_number": 1,
      "action_type": "explore",  // explore | analyze | interpret | decide | verify
      "reasoning": "Before running any models, I need to understand the structure of the data. I have 460 sub-basins with speciation rates and 15+ potential predictors spanning morphological evolution, climate, habitat, and diversity...",
      "action": "Compute summary statistics and correlation matrix for all variables against BAMM_speciation",
      "expected_outcome": "Identify which variables have strongest univariate associations with speciation rate",
      "actual_outcome": "MBL_evol has highest correlation (r=0.45). diversity and area also moderate (r~0.2). Temperature weak (r=0.08)."
    },
    {
      "step_number": 2,
      ...
    }
  ],
  "final_hypothesis": "...",
  "gold_hypothesis": "..."
}
```

## Fork pair format

```json
{
  "task_id": "biology_fish_speciation",
  "fork_step": 3,
  "context": "Steps 1-2 are identical. At step 3, after seeing univariate correlations...",
  "shared_state": "We know MBL_evol has highest correlation, diversity and area are moderate, temperature is weak.",
  "chosen": {
    "source": "privi",
    "reasoning": "Univariate correlations are misleading due to collinearity. Need multiple regression to isolate partial effects of each predictor category.",
    "action": "Run multiple linear regression with all predictors simultaneously. Also run hierarchical partitioning to decompose variance."
  },
  "rejected": {
    "source": "base",
    "reasoning": "Several variables correlate with speciation. Use Random Forest to capture non-linearities and rank feature importance.",
    "action": "Fit Random Forest, extract feature importance scores, report top 3 predictors."
  }
}
```

## Next steps after POC validates

If human evaluators consistently prefer privi choices:

1. **Scale data generation**: Run privi/base on all 264 DiscoveryBench tasks + additional papers with public data
2. **Train with DPO**: Fine-tune a model on the fork pairs
3. **Evaluate**: Test the trained model on held-out DiscoveryBench tasks — does it make better decisions than the base model?
4. **Compare**: Is the DPO-trained model better than SFT on privi trajectories? (This validates the preference learning over pure imitation)
