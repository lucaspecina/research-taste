"""Render an anchored trajectory as readable Markdown.

Usage:
    python src/render_anchored.py --input trajectories/biology_fish/anchored_1.json
    python src/render_anchored.py --input trajectories/*/anchored_*.json
"""

import argparse
import glob
import json
import os


def render_anchored(data):
    """Render an anchored trajectory as Markdown."""
    t = data["privi_trajectory"]
    pairs = data["pairs"]
    score = data.get("score", {})

    lines = []
    lines.append(f"# Anchored Trajectory: {data['task_id']}")
    lines.append("")
    lines.append(f"**Model:** {data['model']} | **Anchor:** {data['anchor']} | **Pairs:** {data['num_pairs']} | **Score:** {score.get('total', 'N/A')}")
    lines.append("")
    lines.append("## Research Question")
    lines.append("")
    lines.append(f"> {t['gold_hypothesis']}")
    lines.append("")

    for i, step in enumerate(t["steps"]):
        pair = pairs[i] if i < len(pairs) else None
        er = step.get("execution_result", {})

        lines.append("---")
        lines.append("")
        lines.append(f"## Step {step['step_number']} -- {step['action_type'].upper()}")
        lines.append("")

        # Privi
        lines.append("### Privi (has paper)")
        lines.append("")
        lines.append(f"**Reasoning:** {step['reasoning']}")
        lines.append("")
        lines.append(f"**Action:** {step['action']}")
        lines.append("")

        # Privi code
        code = step.get("code", "")
        if code.strip():
            lines.append("**Code:**")
            lines.append("")
            lines.append("```python")
            lines.append(code.strip())
            lines.append("```")
            lines.append("")

        # Execution result
        if er.get("exit_code", -1) == 0 and er.get("stdout"):
            stdout = er["stdout"].strip()
            out_lines = stdout.split("\n")
            if len(out_lines) > 30:
                stdout = "\n".join(out_lines[:30]) + f"\n... ({len(out_lines) - 30} more lines)"
            lines.append("**Output:**")
            lines.append("")
            lines.append("```")
            lines.append(stdout)
            lines.append("```")
        elif er.get("exit_code", -1) != 0:
            stderr = er.get("stderr", "").strip()
            lines.append("**Output:** ERROR")
            if stderr:
                lines.append("")
                lines.append("```")
                lines.append(stderr[:500])
                lines.append("```")
        lines.append("")

        # Base counterfactual
        if pair:
            rej = pair["rejected"]
            div = pair["divergence_score"]
            lines.append(f"### Base (no paper) -- same state -- divergence: {div}")
            lines.append("")
            lines.append(f"**Reasoning:** {rej['reasoning']}")
            lines.append("")
            lines.append(f"**Action:** {rej['action']}")
            lines.append("")

            # Base code
            base_code = rej.get("code", "")
            if base_code.strip():
                lines.append("**Code:**")
                lines.append("")
                lines.append("```python")
                lines.append(base_code.strip())
                lines.append("```")
            lines.append("")

    # Final hypothesis
    lines.append("---")
    lines.append("")
    lines.append("## Final Hypothesis")
    lines.append("")
    lines.append(f"**Privi:** {t['final_hypothesis']}")
    lines.append("")
    lines.append(f"**Gold:** {t['gold_hypothesis']}")
    lines.append("")
    lines.append(f"**Score:** {score.get('total', 'N/A')}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Render anchored trajectory as Markdown")
    parser.add_argument("--input", required=True, nargs="+", help="Anchored JSON file(s)")
    args = parser.parse_args()

    input_files = []
    for pattern in args.input:
        input_files.extend(glob.glob(pattern))

    for path in input_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        md = render_anchored(data)

        out_path = path.replace(".json", ".md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)

        print(f"Rendered: {out_path}")


if __name__ == "__main__":
    main()
