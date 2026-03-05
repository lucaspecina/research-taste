"""Format fork pairs for blinded human evaluation.

Randomizes the order of chosen/rejected so evaluators don't know which is privi.

Usage:
    python src/format_eval.py \
        --forks trajectories/biology_fish/forks.json \
        --output eval/biology_fish_pairs.json
"""

import argparse
import json
import os
import random


def load_forks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def blind_pair(fork, pair_id):
    """Create a blinded evaluation pair with randomized order."""
    options = [
        {"label": "A", "reasoning": fork["chosen"]["reasoning"], "action": fork["chosen"]["action"], "_source": "privi"},
        {"label": "B", "reasoning": fork["rejected"]["reasoning"], "action": fork["rejected"]["action"], "_source": "base"},
    ]

    # Randomize order
    if random.random() > 0.5:
        options.reverse()
        options[0]["label"] = "A"
        options[1]["label"] = "B"

    return {
        "pair_id": pair_id,
        "fork_step": fork["fork_step"],
        "context": fork["context"],
        "shared_state": fork["shared_state"],
        "option_a": {
            "reasoning": options[0]["reasoning"],
            "action": options[0]["action"],
        },
        "option_b": {
            "reasoning": options[1]["reasoning"],
            "action": options[1]["action"],
        },
        # Hidden from evaluator, used for scoring
        "_answer_key": {
            "privi_is": options[0]["label"] if options[0]["_source"] == "privi" else options[1]["label"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Format forks for human evaluation")
    parser.add_argument("--forks", required=True, help="Path to forks JSON")
    parser.add_argument("--output", required=True, help="Output eval pairs JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    forks_data = load_forks(args.forks)

    pairs = []
    for i, fork in enumerate(forks_data["forks"]):
        pair = blind_pair(fork, pair_id=f"{forks_data['task_id']}_fork_{i+1}")
        pairs.append(pair)

    output = {
        "task_id": forks_data["task_id"],
        "num_pairs": len(pairs),
        "pairs": pairs,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Eval pairs saved: {args.output}")
    print(f"Total pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
