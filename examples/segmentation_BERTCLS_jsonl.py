import torch
from tqdm import tqdm
import json
import argparse
import numpy as np
import os

from transformers import BertConfig,  BertModel, BertTokenizer
from utils_segmentation import convert_examples_to_features, read_expamples_2
WINDOW_SIZE = 2
SEGMENT_JUMP_STEP = 2
SIMILARITY_THRESHOLD = 0.6
MAX_SEGMENT_ROUND = 6
MAX_SEQ_LENGTH = 50
MODEL_CLASSES = {
    'bert': (BertConfig,  BertModel, BertTokenizer),
}
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def generate_vectors_2(model,examples,tokenizer,device):
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

def segmentation(documents):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
    config = config_class.from_pretrained(args.berttype)
    tokenizer = tokenizer_class.from_pretrained(args.berttype,do_lower_case=True)
    model = model_class.from_pretrained(args.berttype ,config=config).to(device)
    model.eval()

    all_cut_list = []
    pbar = tqdm(documents, total=len(documents), unit="conv", position=0)
    for document_o in pbar:
        if(len(document_o)%2):
            document=document_o[1:]
        else:
            document = document_o
        cut_index=0
        cut_list = []
        inner_pbar = tqdm(total=len(document), unit="msg", position=1, leave=False)
        while(cut_index<len(document)):
            left_sent=""
            i=0
            temp_sent = ""
            final_value=2
            final_cutpoint=len(document)-1

            if(cut_index-WINDOW_SIZE>0):
                index=WINDOW_SIZE
                while(index>0):
                    left_sent+=document[cut_index-index]
                    index-=1

            else:
                temp_index=0
                while(temp_index<cut_index):
                    left_sent+=document[temp_index]
                    temp_index+=1

            while (cut_index + i < len(document) and i < MAX_SEGMENT_ROUND):
                temp_sent+=document[cut_index+i]
                if(i%SEGMENT_JUMP_STEP==SEGMENT_JUMP_STEP-1):
                    bert_input=[]
                    right_sent=""
                    if(cut_index+i+WINDOW_SIZE<len(document)):
                        index=1
                        while(index<=WINDOW_SIZE):
                            right_sent+=document[cut_index+i+index]
                            index+=1
                    else:
                        temp_index=1
                        while(cut_index+i+temp_index<len(document)):
                            right_sent += document[cut_index + i + temp_index]
                            temp_index+=1

                    if(left_sent):
                        bert_input.append(left_sent)
                    bert_input.append(temp_sent)
                    if(right_sent):
                        bert_input.append(right_sent)
                    examples=read_expamples_2(bert_input)
                    vectors=generate_vectors_2(model,examples,tokenizer,device)
                    if(left_sent):
                        left_value=similarity(vectors[0],vectors[1])
                        right_value = similarity(vectors[1], vectors[2]) if right_sent else -1
                    else:
                        left_value=-1
                        right_value=similarity(vectors[0],vectors[1]) if right_sent else -1
                    larger_value=left_value if left_value > right_value else right_value
                    if(not left_sent and not right_sent):
                        larger_value=SIMILARITY_THRESHOLD
                    if(larger_value<final_value):
                        final_value=larger_value
                        final_cutpoint=cut_index + i
                i+=1

            cut_list.append(final_cutpoint)
            prev_cut_index = cut_index
            cut_index=final_cutpoint+1
            inner_pbar.update(cut_index - prev_cut_index)
        inner_pbar.close()
        if(len(document_o)%2):
            cut_list_new=[i+1 for i in cut_list]
        else:
            cut_list_new=cut_list
        if (len(cut_list) == 0):
            cut_list_new = [0]
        assert cut_list_new[-1] == len(document_o) - 1
        all_cut_list.append(cut_list_new)
    pbar.close()
    return all_cut_list


def read_jsonl_conversations(jsonl_path):
    """Read a JSONL file where each line is {"author": "...", "content": "..."}.
    Returns a list of (turns, author_map) where:
      - turns: list of {"author": str, "content": str}
      - author_map: {author_name: "Person_1" or "Person_2"} by first-appearance order
    Each JSONL file is treated as a single conversation.
    """
    turns = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))

    # Map authors to Person_1 / Person_2 by order of first appearance
    author_map = {}
    for turn in turns:
        author = turn["author"]
        if author not in author_map:
            if not author_map:
                author_map[author] = "Person_1"
            elif len(author_map) == 1:
                author_map[author] = "Person_2"
            # More than 2 authors: keep existing roles, extras are unmapped

    return turns, author_map


# python segmentation_BERTCLS_jsonl.py --input ../data/pairs_grouped.jsonl --output ../data/pairs_grouped_topics.jsonl
def segment_jsonl_file(input_jsonl_path, output_jsonl_path):
    turns, author_map = read_jsonl_conversations(input_jsonl_path)

    # Build the flat document (list of content strings) for the segmentation algorithm
    document = [turn["content"] for turn in turns]

    all_cut_list = segmentation([document])
    cut_list = all_cut_list[0]

    segments = []
    prev = 0
    msg_index = 0
    for cut in cut_list:
        seg_turns = turns[prev:cut + 1]

        # If this segment would start on Person_2 (odd msg_index),
        # move that first turn to the previous segment so every
        # segment starts with Person_1.
        if msg_index % 2 == 1 and segments:
            first_turn = seg_turns[0]
            role = author_map.get(first_turn["author"], "Person_2")
            segments[-1]["messages"].append({"role": role, "content": first_turn["content"]})
            msg_index += 1
            seg_turns = seg_turns[1:]

        messages = []
        for turn in seg_turns:
            role = author_map.get(turn["author"], f"Person_{msg_index % 2 + 1}")
            messages.append({"role": role, "content": turn["content"]})
            msg_index += 1

        if messages:
            segments.append({"segment_id": len(segments), "messages": messages})
        prev = cut + 1

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + '\n')
    print("Done. Segmented output saved to", output_jsonl_path)


parser = argparse.ArgumentParser()
parser.add_argument("--berttype",
                    default='bert-base-uncased',
                    type=str,
                    help="The type of BERT")
parser.add_argument("--input",
                    default=None,
                    type=str,
                    help="Path to input JSONL file (author/content turns) to segment")
parser.add_argument("--output",
                    default=None,
                    type=str,
                    help="Path to output JSONL file for segmented results")
args = parser.parse_args()


if __name__ == '__main__':
    if args.input and args.output:
        segment_jsonl_file(args.input, args.output)
    else:
        print("Please provide --input and --output paths.")
        print("Example: python segmentation_BERTCLS_jsonl.py --input ../data/pairs_grouped.jsonl --output ../data/pairs_grouped_topics.jsonl")
