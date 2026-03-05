"""Shared utilities for all scripts."""

import json
import os

import pandas as pd


def load_task(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_paper(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompt(name):
    path = os.path.join("prompts", name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_dataset_summary(task):
    """Build a text summary of datasets, including a preview if CSV exists."""
    parts = []
    for ds in task["datasets"]:
        part = f"Dataset: {ds['name']}\n"
        part += f"Description: {ds['description']}\n"
        if ds.get("columns"):
            part += "Columns:\n"
            for col_name, col_desc in ds["columns"].items():
                part += f"  - {col_name}: {col_desc}\n"
        if ds.get("csv_path") and os.path.exists(ds["csv_path"]):
            try:
                df = pd.read_csv(ds["csv_path"])
                part += f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
                part += f"Preview (first 5 rows):\n{df.head().to_string()}\n"
                part += f"Summary statistics:\n{df.describe().to_string()}\n"
            except Exception as e:
                part += f"(Could not load CSV: {e})\n"
        parts.append(part)
    return "\n---\n".join(parts)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_steps(steps):
    """Format previous steps as text for prompts."""
    if not steps:
        return "(No steps yet -- this is the first step.)"
    parts = []
    for s in steps:
        parts.append(f"Step {s['step_number']} [{s['action_type']}]: {s['action']}\n  Outcome: {s['actual_outcome']}")
    return "\n".join(parts)
