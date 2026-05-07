"""
Post-process a segments JSONL file to fix ordering issues caused by segments
that end on Person_1.

For every segment that ends with a Person_1 message:
  - Remove that trailing Person_1 message from the current segment.
  - Remove the first message of the immediately following segment.

Segments that become empty after trimming are dropped.
Segment IDs are reassigned sequentially after all edits.

Usage:
    python fix_segment_order.py input.jsonl output.jsonl
"""

import json
import glob
import os
import sys

INPUT_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'transformed', 'plain', 'plain_pairs'))
OUTPUT_DIR = os.path.join(INPUT_DIR, 'fixed')


def load_jsonl(path):
    segments = []
    with open(path, "r", encoding="utf-8") as reader:
        for lineno, line in enumerate(reader, 1):
            line = line.strip()
            if not line:
                continue
            try:
                segments.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {lineno}: {e}", file=sys.stderr)
    return segments

def checking():
    jsonl_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.jsonl')))
    for i, path in enumerate(jsonl_files):
        if i<4:
            segments = load_jsonl(path)
            segment_list = [list(seg["messages"]) for seg in segments]
            print(segment_list)

def fix_segments(segments):
    segment_list = [list(seg["messages"]) for seg in segments]

    i = 0
    while i < len(segment_list):
        msgs = segment_list[i]
        if msgs and msgs[-1]["role"] == "Person_1":
            # Remove the trailing Person_1 message from this segment.
            segment_list[i] = msgs[:-1]
            # Remove the first message of the next segment (if one exists).
            if i + 1 < len(segment_list) and segment_list[i + 1]:
                segment_list[i + 1] = segment_list[i + 1][1:]
        i += 1

    # Rebuild segments, dropping any that are now empty, and reassign IDs.
    result = []
    for msgs in segment_list:
        if msgs:
            result.append({"segment_id": len(result), "messages": msgs})

    return result

def main():
# if __name__ == "__main__":
    jsonl_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.jsonl')))
    # if not jsonl_files:
    #     print(f"No .jsonl files found in {INPUT_DIR}")
    #     sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path in jsonl_files:
        segments = load_jsonl(path)
        fixed = fix_segments(segments)
        dropped = len(segments) - len(fixed)

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
        with open(out_path, "w", encoding="utf-8") as f:
            for seg in fixed:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")

        print(f"{os.path.basename(path)}: {len(segments)} -> {len(fixed)} segments ({dropped} dropped)")

    print("Done.")

main()
# checking()