"""Shared semi-agentic generation loop.

Each step: LLM proposes analysis + writes code -> we execute on real data -> feed results back.
Used by generate_privi.py and generate_base.py with --mode agentic.
"""

from common import build_dataset_summary, build_df_description
from llm import call_messages, extract_json, get_model
from sandbox import execute_code


def _format_history(steps):
    """Format completed steps + their execution results for the prompt."""
    if not steps:
        return "(No steps yet — this is the first step.)"
    parts = []
    for s in steps:
        part = f"### Step {s['step_number']} [{s['action_type']}]\n"
        part += f"**Reasoning:** {s['reasoning']}\n"
        part += f"**Action:** {s['action']}\n"
        part += f"**Expected:** {s['expected_outcome']}\n"
        er = s.get("execution_result", {})
        if er.get("exit_code", 0) == 0 and er.get("stdout"):
            part += f"**Result (real data):**\n```\n{er['stdout']}\n```"
        elif er.get("stderr"):
            part += f"**Error:**\n```\n{er['stderr']}\n```"
        else:
            part += f"**Result:** {s.get('actual_outcome', '(no output)')}"
        parts.append(part)
    return "\n\n".join(parts)


def _build_step_prompt(task, step_number, steps_so_far):
    """Build the user prompt for one step."""
    query = task["queries"][0]
    df_desc = build_df_description(task)
    history = _format_history(steps_so_far)

    return f"""You are on step {step_number} of your investigation.

## Research Question
{query['question']}

## Available DataFrames (pre-loaded)
{df_desc}

## Previous Steps and Results
{history}

## Instructions
Propose the next research step. Return ONLY a JSON object:

```json
{{
  "step_number": {step_number},
  "action_type": "explore|analyze|interpret|decide|verify",
  "reasoning": "Why this step makes sense given what you've seen so far...",
  "action": "Natural language description of what to do...",
  "code": "Python code that performs the analysis. Print results to stdout.",
  "expected_outcome": "What you expect to find before running the code..."
}}
```

IMPORTANT:
- Your code will be executed on REAL data. You will see real results.
- Available: pandas (pd), numpy (np), scipy.stats, sklearn, statsmodels.
- DataFrames are already loaded — do NOT load them yourself.
- Always print() results. Keep output concise.
- Do NOT produce plots. Describe findings via print() instead."""


async def generate_trajectory_loop(task, system_prompt, paper_text, num_steps, max_consecutive_errors=3):
    """Run the semi-agentic loop: LLM proposes -> execute -> feed back -> repeat."""
    model = get_model()
    dataset_summary = build_dataset_summary(task)
    query = task["queries"][0]
    domain_knowledge = task.get("domain_knowledge") or f"Domain: {task['domain']}"

    # Build initial system + context message
    context = f"""## Research Question
{query['question']}

## Domain Context
{domain_knowledge}

## Available Data
{dataset_summary}
"""
    if paper_text:
        context += f"\n## Published Paper (PRIVILEGED INFORMATION)\n{paper_text}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context + "\nI will now ask you to generate steps one at a time. For each step, propose an analysis AND write executable Python code. I will run your code on the real data and show you the results."},
    ]

    steps = []
    consecutive_errors = 0

    for step_num in range(1, num_steps + 1):
        print(f"  Step {step_num}/{num_steps}...")

        # Ask for this step
        step_prompt = _build_step_prompt(task, step_num, steps)
        messages.append({"role": "user", "content": step_prompt})

        # Call LLM
        text = await call_messages(messages, max_tokens=4096)
        messages.append({"role": "assistant", "content": text})

        # Parse step JSON
        step = extract_json(text)
        if step is None:
            print(f"    WARNING: could not parse JSON for step {step_num}, skipping")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"    ABORT: {max_consecutive_errors} consecutive errors")
                break
            continue

        # Execute code
        code = step.get("code", "")
        if code:
            print(f"    Executing code...")
            er = execute_code(code, task)
            step["execution_result"] = er

            if er["exit_code"] == 0:
                step["actual_outcome"] = er["stdout"].strip()
                consecutive_errors = 0
                print(f"    OK ({len(er['stdout'])} chars output)")
            else:
                step["actual_outcome"] = f"ERROR: {er['stderr'][:500]}"
                consecutive_errors += 1
                print(f"    ERROR: {er['stderr'][:200]}")

            # Feed execution result back
            result_msg = f"## Execution Result for Step {step_num}\n"
            if er["exit_code"] == 0:
                result_msg += f"```\n{er['stdout']}\n```"
            else:
                result_msg += f"Error (exit code {er['exit_code']}):\n```\n{er['stderr']}\n```"
            messages.append({"role": "user", "content": result_msg})
        else:
            step["execution_result"] = {}
            step["actual_outcome"] = "(no code to execute)"
            print(f"    No code provided")

        steps.append(step)

        if consecutive_errors >= max_consecutive_errors:
            print(f"    ABORT: {max_consecutive_errors} consecutive errors")
            break

    # Final hypothesis
    print(f"  Generating final hypothesis...")
    messages.append({"role": "user", "content": "Based on all the analyses you've run and their real results, state your final hypothesis answering the research question. Return ONLY a JSON object: {\"final_hypothesis\": \"...\"}"})
    text = await call_messages(messages, max_tokens=1024)
    final = extract_json(text)
    final_hypothesis = final.get("final_hypothesis", "") if final else ""

    return {"steps": steps, "final_hypothesis": final_hypothesis}
