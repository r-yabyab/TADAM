"""
Scan a segments JSONL file and log which line numbers contain a segment
whose last message has role == "Person_1".

Usage:
    python log_trailing_person1.py          (scans INPUT_DIR)
"""

import json
import glob
import os
import sys

INPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'transformed', 'plain', 'plain_pairs'))


def log_trailing_person1(path):
    hits = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                seg = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {lineno}: {e}", file=sys.stderr)
                continue
            msgs = seg.get("messages", [])
            if msgs and msgs[-1].get("role") == "Person_1":
                hits.append(lineno)
    return hits


def main():
    jsonl_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.jsonl')))

    for path in jsonl_files:
        hits = log_trailing_person1(path)
        if hits:
            print(f"{os.path.basename(path)}: trailing Person_1 on lines {hits}")
        else:
            print(f"{os.path.basename(path)}: no trailing Person_1 found")

    print("Done.")

main()
