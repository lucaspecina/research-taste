"""Generate interleaved trajectories where base and privi alternate steps.

The base generates a step, then the privi continues from that state.
This creates more diverse training data by forcing the privi to recover
from the base's (potentially suboptimal) decisions.

Usage:
    python src/generate_interleaved.py \
        --task data/tasks/biology_fish.json \
        --paper data/papers/cerezer2023.txt \
        --output trajectories/biology_fish/interleaved_1.json \
        --steps 6
"""

import argparse
import asyncio

from common import load_task, load_paper, load_prompt, build_dataset_summary, format_steps, save_json
from llm import call, extract_json, get_model


async def generate_step(system_prompt, task, dataset_summary, paper_text, steps_so_far, step_number):
    query = task["queries"][0]
    domain_knowledge = task.get("domain_knowledge") or f"Domain: {task['domain']}"

    prompt = f"""## Research Question
{query['question']}

## Domain Context
{domain_knowledge}

## Available Data
{dataset_summary}
"""
    if paper_text:
        prompt += f"\n## Published Paper (PRIVILEGED INFORMATION)\n{paper_text}\n"

    prompt += f"""
## Steps so far
{format_steps(steps_so_far)}

## Instructions
Generate step {step_number}. Return ONLY a JSON object:

```json
{{
  "step_number": {step_number},
  "action_type": "explore|analyze|interpret|decide|verify",
  "reasoning": "...",
  "action": "...",
  "expected_outcome": "...",
  "actual_outcome": "..."
}}
```
"""

    text = await call(prompt, system_prompt=system_prompt, max_tokens=2048)
    result = extract_json(text)
    if result is None:
        raise ValueError(f"Failed to parse JSON from LLM response:\n{text[:500]}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate interleaved trajectory")
    parser.add_argument("--task", required=True, help="Path to task JSON")
    parser.add_argument("--paper", required=True, help="Path to paper text file")
    parser.add_argument("--output", required=True, help="Output trajectory JSON path")
    parser.add_argument("--steps", type=int, default=6, help="Number of steps (5-8)")
    args = parser.parse_args()

    task = load_task(args.task)
    paper_text = load_paper(args.paper)
    dataset_summary = build_dataset_summary(task)

    privi_system = load_prompt("privi_system.txt")
    base_system = load_prompt("base_system.txt")

    async def run():
        steps = []
        for i in range(1, args.steps + 1):
            is_base_turn = (i % 2 == 1)  # odd steps = base, even steps = privi
            mode = "base" if is_base_turn else "privi"
            system = base_system if is_base_turn else privi_system
            paper = None if is_base_turn else paper_text

            print(f"Step {i}/{args.steps} ({mode})...")
            step = await generate_step(system, task, dataset_summary, paper, steps, i)
            step["generated_by"] = mode
            steps.append(step)
        return steps

    steps = asyncio.run(run())

    trajectory = {
        "task_id": task["task_id"],
        "model": get_model(),
        "mode": "interleaved",
        "paper_in_context": True,
        "steps": steps,
        "gold_hypothesis": task["queries"][0].get("true_hypothesis", ""),
    }

    save_json(trajectory, args.output)
    print(f"Interleaved trajectory saved: {args.output}")
    for s in steps:
        print(f"  Step {s['step_number']} [{s['generated_by']}]: {s['action'][:60]}...")


if __name__ == "__main__":
    main()
