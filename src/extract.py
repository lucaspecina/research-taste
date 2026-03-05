"""Extract task info from DiscoveryBench metadata into a structured JSON task file.

Usage:
    python src/extract.py --task biology_fish --output data/tasks/biology_fish.json
    python src/extract.py --task sociology_bmi --output data/tasks/sociology_bmi.json
    python src/extract.py --task economics_immigration --output data/tasks/economics_immigration.json
"""

import argparse
import json
import os
import glob

DISCOVERYBENCH_ROOT = os.path.join("data", "discoverybench", "discoverybench", "real")


def find_metadata_files(domain_path):
    """Find all metadata JSON files in a domain folder."""
    pattern = os.path.join(domain_path, "metadata_*.json")
    return sorted(glob.glob(pattern))


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_csv_files(domain_path):
    """Find all CSV files in the domain's dataset folders."""
    csvs = {}
    for dirpath, _, filenames in os.walk(domain_path):
        for fn in filenames:
            if fn.endswith(".csv"):
                csvs[fn] = os.path.join(dirpath, fn)
    return csvs


def extract_task(metadata, domain_path, domain_name):
    """Extract a clean task object from DiscoveryBench metadata."""
    datasets_info = []
    available_csvs = find_csv_files(domain_path)

    for ds in metadata.get("datasets", []):
        ds_name = ds.get("name", "")
        ds_entry = {
            "name": ds_name,
            "description": ds.get("description", ""),
            "columns": ds.get("columns", {}),
        }
        # Try to resolve the actual CSV path
        if ds_name in available_csvs:
            ds_entry["csv_path"] = available_csvs[ds_name]
        datasets_info.append(ds_entry)

    queries = []
    for q in metadata.get("queries", []):
        queries.append({
            "question": q.get("question", ""),
            "true_hypothesis": q.get("true_hypothesis", {}).get("hypothesis", q.get("true_hypothesis", "")),
        })

    return {
        "task_id": f"{domain_name}_{metadata.get('metadata_id', 'unknown')}",
        "domain": domain_name,
        "workflow_tags": metadata.get("workflow_tags", []),
        "domain_knowledge": metadata.get("domain_knowledge", None),
        "datasets": datasets_info,
        "queries": queries,
    }


# Mapping of our POC task aliases to DiscoveryBench domain/keyword
TASK_ALIASES = {
    "biology_fish": {
        "domain": "biology",
        "keyword": "body-size-evolution",
    },
    "sociology_bmi": {
        "domain": "sociology",
        "keyword": "nls_bmi",
    },
    "economics_immigration": {
        "domain": "economics",
        "keyword": "offshoring",
    },
}


def find_matching_metadata(domain_path, keyword):
    """Find the metadata file whose datasets match a keyword."""
    for meta_path in find_metadata_files(domain_path):
        meta = load_metadata(meta_path)
        meta_str = json.dumps(meta).lower()
        if keyword.lower() in meta_str:
            return meta_path, meta
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Extract task from DiscoveryBench")
    parser.add_argument("--task", required=True, choices=list(TASK_ALIASES.keys()),
                        help="Task alias (biology_fish, sociology_bmi, economics_immigration)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--discoverybench-root", default=DISCOVERYBENCH_ROOT,
                        help="Root path to DiscoveryBench real tasks")
    args = parser.parse_args()

    alias = TASK_ALIASES[args.task]
    domain = alias["domain"]
    keyword = alias["keyword"]

    domain_path = os.path.join(args.discoverybench_root, domain)
    if not os.path.isdir(domain_path):
        print(f"ERROR: Domain path not found: {domain_path}")
        print(f"Make sure you cloned DiscoveryBench: git clone https://github.com/allenai/discoverybench data/discoverybench")
        return

    meta_path, meta = find_matching_metadata(domain_path, keyword)
    if meta is None:
        print(f"ERROR: No metadata found matching keyword '{keyword}' in {domain_path}")
        print("Available metadata files:")
        for p in find_metadata_files(domain_path):
            print(f"  {p}")
        return

    print(f"Found metadata: {meta_path}")
    task = extract_task(meta, domain_path, domain)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)

    print(f"Task extracted: {task['task_id']}")
    print(f"  Domain: {task['domain']}")
    print(f"  Datasets: {len(task['datasets'])}")
    print(f"  Queries: {len(task['queries'])}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
