"""Extract fork pairs from privi and base trajectories.

Uses an LLM to compare trajectories and identify meaningful divergence points.

Usage:
    python src/extract_forks.py \
        --privi trajectories/biology_fish/privi_1.json \
        --base trajectories/biology_fish/base_1.json \
        --output trajectories/biology_fish/forks.json
"""

import argparse
import asyncio
import json

from common import load_task, save_json
from llm import call, extract_json, get_model


def load_trajectory(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def extract_forks_via_llm(privi_traj, base_traj):
    privi_steps = json.dumps(privi_traj["steps"], indent=2)
    base_steps = json.dumps(base_traj["steps"], indent=2)

    prompt = f"""Compare these two research trajectories for the same task and identify fork points -- steps where the researchers made meaningfully different decisions.

## Privileged trajectory (has paper context)
{privi_steps}

## Base trajectory (no paper)
{base_steps}

## Instructions
Identify 2-4 fork points where the trajectories diverge in a meaningful way. For each fork:
1. Identify the step number where divergence occurs
2. Describe the shared state (what both researchers knew at that point)
3. Extract the chosen (privi) and rejected (base) decisions

Return as JSON:
```json
{{
  "forks": [
    {{
      "fork_step": 3,
      "context": "At step 3, after exploring the data...",
      "shared_state": "Both researchers know that...",
      "chosen": {{
        "source": "privi",
        "reasoning": "...",
        "action": "..."
      }},
      "rejected": {{
        "source": "base",
        "reasoning": "...",
        "action": "..."
      }}
    }}
  ]
}}
```

Focus on forks where the DECISION quality differs, not just surface-level differences in wording.
"""

    model = get_model()
    print(f"Analyzing trajectories with {model}...")
    text = await call(prompt)
    result = extract_json(text)
    if result is None:
        raise ValueError(f"Failed to parse JSON from LLM response:\n{text[:500]}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract fork pairs from trajectories")
    parser.add_argument("--privi", required=True, help="Path to privi trajectory JSON")
    parser.add_argument("--base", required=True, help="Path to base trajectory JSON")
    parser.add_argument("--output", required=True, help="Output forks JSON path")
    args = parser.parse_args()

    privi_traj = load_trajectory(args.privi)
    base_traj = load_trajectory(args.base)

    result = asyncio.run(extract_forks_via_llm(privi_traj, base_traj))

    output = {
        "task_id": privi_traj["task_id"],
        "privi_source": args.privi,
        "base_source": args.base,
        "forks": result["forks"],
    }

    save_json(output, args.output)
    print(f"Forks saved: {args.output}")
    print(f"Found {len(output['forks'])} fork points")
    for fork in output["forks"]:
        print(f"  Step {fork['fork_step']}: {fork['context'][:80]}...")


if __name__ == "__main__":
    main()
