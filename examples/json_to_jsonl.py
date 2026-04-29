"""
Convert a segmented JSON file (conversations → segments → messages)
to JSONL with one segment per line.

Usage:
    python json_to_jsonl.py --input ../data/pairs_plain_topics.json --output ../data/pairs_plain_topics.jsonl
"""

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str, help="Path to segmented JSON file")
parser.add_argument("--output", required=True, type=str, help="Path to output JSONL file")
args = parser.parse_args()

with open(args.input, 'r', encoding='utf-8') as f:
    segmented_documents = json.load(f)

total = 0
with open(args.output, 'w', encoding='utf-8') as f:
    for conv_id, segments in enumerate(segmented_documents):
        for seg_id, segment in enumerate(segments):
            line = {"conversation_id": conv_id, "segment_id": seg_id, "messages": segment}
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
            total += 1

print(f"Done. {total} segments written to {args.output}")
