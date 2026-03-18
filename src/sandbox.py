"""Execute model-generated Python code on real data via subprocess."""

import os
import subprocess
import sys
import tempfile


MAX_OUTPUT = 3000  # truncate stdout to this many chars

# Project root: one level up from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_preamble(task):
    """Generate Python code that loads the task's datasets into DataFrames."""
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "pd.set_option('display.max_columns', None)",
        "pd.set_option('display.width', 200)",
        "",
    ]

    datasets = task.get("datasets", [])
    for i, ds in enumerate(datasets):
        raw_path = ds.get("csv_path", "")
        abs_path = os.path.join(PROJECT_ROOT, raw_path).replace("\\", "/")
        var_name = "df" if len(datasets) == 1 else f"df_{i + 1}"

        if raw_path.endswith(".dta"):
            lines.append(f'{var_name} = pd.read_stata("{abs_path}")  # {ds["name"]}')
        else:
            lines.append(f'{var_name} = pd.read_csv("{abs_path}")  # {ds["name"]}')

    lines.append("")
    return "\n".join(lines)


def sanitize_code(code):
    """Fix common LLM code issues: literal \\n escapes, redundant imports."""
    # The model sometimes emits escaped newlines as literal \n inside a JSON string
    if "\\n" in code and "\n" not in code.replace("\\n", ""):
        code = code.replace("\\n", "\n")
    # Remove redundant imports (already in preamble)
    lines = code.split("\n")
    lines = [l for l in lines if not l.strip().startswith(("import pandas", "import numpy", "import warnings"))]
    return "\n".join(lines)


def execute_code(code, task, timeout=60):
    """Run model-generated code with pre-loaded data. Returns dict with stdout/stderr/exit_code."""
    code = sanitize_code(code)
    preamble = build_preamble(task)
    full_code = preamble + "\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        stdout = result.stdout
        truncated = len(stdout) > MAX_OUTPUT
        if truncated:
            stdout = stdout[:MAX_OUTPUT] + "\n[TRUNCATED]"

        return {
            "stdout": stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "truncated": truncated,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "exit_code": -1,
            "truncated": False,
        }
    finally:
        os.unlink(tmp_path)
