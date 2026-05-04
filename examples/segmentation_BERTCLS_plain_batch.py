import torch
from tqdm import tqdm
import json
import argparse
import numpy as np
import os
import glob

from transformers import BertConfig, BertModel, BertTokenizer
from utils_segmentation import convert_examples_to_features, read_expamples_2

"""
main one, .json (token size filtered) to .jsonl with segments (to be filtered)
"""


WINDOW_SIZE = 2
SEGMENT_JUMP_STEP = 2
SIMILARITY_THRESHOLD = 0.6
MAX_SEGMENT_ROUND = 6
MAX_SEQ_LENGTH = 50
MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
}
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def generate_vectors_2(model, examples, tokenizer, device):
    features = convert_examples_to_features(examples, MAX_SEQ_LENGTH, tokenizer,
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=0,
                                            sep_token=tokenizer.sep_token,
                                            )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(device)

    vectors = model(input_ids=all_input_ids, attention_mask=all_input_mask, token_type_ids=all_segment_ids)[1]
    return vectors.cpu().detach().numpy()


def segmentation(documents, model, tokenizer, device):
    all_cut_list = []
    pbar = tqdm(documents, total=len(documents), unit="conv", position=0)
    for document_o in pbar:
        if len(document_o) % 2:
            document = document_o[1:]
        else:
            document = document_o
        cut_index = 0
        cut_list = []
        inner_pbar = tqdm(total=len(document), unit="msg", position=1, leave=False)
        while cut_index < len(document):
            left_sent = ""
            i = 0
            temp_sent = ""
            final_value = 2
            final_cutpoint = len(document) - 1

            if cut_index - WINDOW_SIZE > 0:
                index = WINDOW_SIZE
                while index > 0:
                    left_sent += document[cut_index - index]
                    index -= 1
            else:
                temp_index = 0
                while temp_index < cut_index:
                    left_sent += document[temp_index]
                    temp_index += 1

            while cut_index + i < len(document) and i < MAX_SEGMENT_ROUND:
                temp_sent += document[cut_index + i]
                if i % SEGMENT_JUMP_STEP == SEGMENT_JUMP_STEP - 1:
                    bert_input = []
                    right_sent = ""
                    if cut_index + i + WINDOW_SIZE < len(document):
                        index = 1
                        while index <= WINDOW_SIZE:
                            right_sent += document[cut_index + i + index]
                            index += 1
                    else:
                        temp_index = 1
                        while cut_index + i + temp_index < len(document):
                            right_sent += document[cut_index + i + temp_index]
                            temp_index += 1

                    if left_sent:
                        bert_input.append(left_sent)
                    bert_input.append(temp_sent)
                    if right_sent:
                        bert_input.append(right_sent)
                    examples = read_expamples_2(bert_input)
                    vectors = generate_vectors_2(model, examples, tokenizer, device)
                    if left_sent:
                        left_value = similarity(vectors[0], vectors[1])
                        right_value = similarity(vectors[1], vectors[2]) if right_sent else -1
                    else:
                        left_value = -1
                        right_value = similarity(vectors[0], vectors[1]) if right_sent else -1
                    larger_value = left_value if left_value > right_value else right_value
                    if not left_sent and not right_sent:
                        larger_value = SIMILARITY_THRESHOLD
                    if larger_value < final_value:
                        final_value = larger_value
                        final_cutpoint = cut_index + i
                i += 1

            cut_list.append(final_cutpoint)
            prev_cut_index = cut_index
            cut_index = final_cutpoint + 1
            inner_pbar.update(cut_index - prev_cut_index)
        inner_pbar.close()

        if len(document_o) % 2:
            cut_list_new = [i + 1 for i in cut_list]
        else:
            cut_list_new = cut_list
        if len(cut_list) == 0:
            cut_list_new = [0]
        assert cut_list_new[-1] == len(document_o) - 1
        all_cut_list.append(cut_list_new)
    pbar.close()
    return all_cut_list


def segment_file(input_json_path, output_jsonl_path, model, tokenizer, device):
    """Segment a single plain JSON conversation file and write pairs_grouped_topics-style JSONL."""
    with open(input_json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Each plain JSON file contains one conversation: [[msg1, msg2, ...]]
    all_cut_list = segmentation(documents, model, tokenizer, device)

    with open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
        for document, cut_list in zip(documents, all_cut_list):
            segments = []
            prev = 0
            msg_index = 0
            for cut in cut_list:
                seg_msgs = document[prev:cut + 1]

                # If this segment would start on Person_2 (odd msg_index),
                # move that first message to the previous segment so every
                # segment starts with Person_1.
                if msg_index % 2 == 1 and segments:
                    first_msg = seg_msgs[0]
                    segments[-1]["messages"].append({"role": "Person_2", "content": first_msg})
                    msg_index += 1
                    seg_msgs = seg_msgs[1:]

                messages = []
                for msg in seg_msgs:
                    messages.append({
                        "role": "Person_1" if msg_index % 2 == 0 else "Person_2",
                        "content": msg,
                    })
                    msg_index += 1

                if messages:
                    segments.append({"segment_id": len(segments), "messages": messages})
                prev = cut + 1

            for seg in segments:
                out_f.write(json.dumps(seg, ensure_ascii=False) + '\n')

    print(f"  -> {output_jsonl_path}")


parser = argparse.ArgumentParser()
parser.add_argument("--berttype",
                    default='bert-base-uncased',
                    type=str,
                    help="The type of BERT model to use")
parser.add_argument("--input_dir",
                    default=None,
                    type=str,
                    help="Directory containing plain JSON files to process (default: ../data/transformed/plain)")
args = parser.parse_args()


if __name__ == '__main__':
    input_dir = args.input_dir or os.path.join(os.path.dirname(__file__), '..', 'data', 'transformed', 'plain')
    input_dir = os.path.normpath(input_dir)

    json_files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    if not json_files:
        print(f"No .json files found in {input_dir}")
        exit(1)

    print(f"Found {len(json_files)} file(s) in {input_dir}")

    # Load model once and reuse across all files
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
    config = config_class.from_pretrained(args.berttype)
    tokenizer = tokenizer_class.from_pretrained(args.berttype, do_lower_case=True)
    model = model_class.from_pretrained(args.berttype, config=config).to(device)
    model.eval()

    output_dir = os.path.join(input_dir, 'plain_pairs')
    os.makedirs(output_dir, exist_ok=True)

    for json_path in json_files:
        base = os.path.splitext(os.path.basename(json_path))[0]  # strip .json
        output_path = os.path.join(output_dir, base + '_topics.jsonl')
        print(f"Processing: {os.path.basename(json_path)}")
        segment_file(json_path, output_path, model, tokenizer, device)

    print("All done.")
