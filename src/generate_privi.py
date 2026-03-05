"""Generate a privileged trajectory for a task.

The privi model receives: research question + dataset info + full paper text.
It generates a 5-8 step research trajectory via Azure OpenAI.

Usage:
    python src/generate_privi.py \
        --task data/tasks/biology_fish.json \
        --paper data/papers/cerezer2023.txt \
        --output trajectories/biology_fish/privi_1.json \
        --steps 6
"""

import argparse
import asyncio

from common import load_task, load_paper, load_prompt, build_dataset_summary, save_json
from llm import call, extract_json, get_model


async def generate_trajectory(task, paper_text, num_steps):
    system_prompt = load_prompt("privi_system.txt")
    dataset_summary = build_dataset_summary(task)
    query = task["queries"][0]
    domain_knowledge = task.get("domain_knowledge") or f"Domain: {task['domain']}"

    prompt = f"""## Research Question
{query['question']}

## Domain Context
{domain_knowledge}

## Available Data
{dataset_summary}

## Published Paper (PRIVILEGED INFORMATION)
{paper_text}

## Instructions
Generate a complete research trajectory of {num_steps} steps to answer the research question.
Return your response as a JSON object with this structure:

```json
{{
  "steps": [
    {{
      "step_number": 1,
      "action_type": "explore|analyze|interpret|decide|verify",
      "reasoning": "...",
      "action": "...",
      "expected_outcome": "...",
      "actual_outcome": "..."
    }}
  ],
  "final_hypothesis": "Your final answer to the research question"
}}
```

Remember: your decisions must be justifiable WITHOUT the paper. Use the paper to make better choices, not to reveal you have it.
"""

    model = get_model()
    print(f"Calling {model} (privi)...")
    text = await call(prompt, system_prompt=system_prompt)
    result = extract_json(text)
    if result is None:
        raise ValueError(f"Failed to parse JSON from LLM response:\n{text[:500]}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate privileged trajectory")
    parser.add_argument("--task", required=True, help="Path to task JSON")
    parser.add_argument("--paper", required=True, help="Path to paper text file")
    parser.add_argument("--output", required=True, help="Output trajectory JSON path")
    parser.add_argument("--steps", type=int, default=6, help="Number of steps (5-8)")
    args = parser.parse_args()

    task = load_task(args.task)
    paper_text = load_paper(args.paper)

    result = asyncio.run(generate_trajectory(task, paper_text, args.steps))

    trajectory = {
        "task_id": task["task_id"],
        "model": get_model(),
        "mode": "privi",
        "paper_in_context": True,
        "steps": result["steps"],
        "final_hypothesis": result.get("final_hypothesis", ""),
        "gold_hypothesis": task["queries"][0].get("true_hypothesis", ""),
    }

    save_json(trajectory, args.output)
    print(f"Trajectory saved: {args.output}")
    print(f"Steps: {len(trajectory['steps'])}")
    print(f"Final hypothesis: {trajectory['final_hypothesis'][:100]}...")


if __name__ == "__main__":
    main()
