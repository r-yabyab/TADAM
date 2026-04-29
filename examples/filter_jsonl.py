"""
Remove JSONL lines where the segment contains no Person_2 message.

Usage:
    python filter_jsonl.py --input ../data/pairs_plain_topics.jsonl --output ../data/pairs_plain_topics_filtered.jsonl
"""

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str, help="Path to input JSONL file")
parser.add_argument("--output", required=True, type=str, help="Path to output JSONL file")
args = parser.parse_args()

kept = 0
removed = 0

with open(args.input, 'r', encoding='utf-8') as infile, \
     open(args.output, 'w', encoding='utf-8') as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        has_person2 = any(msg.get("role") == "Person_2" for msg in obj.get("messages", []))
        if has_person2:
            outfile.write(line + '\n')
            kept += 1
        else:
            removed += 1

print(f"Done. Kept: {kept}, Removed: {removed} → {args.output}")
