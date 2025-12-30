# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import logging
import pickle
import json

import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from torch import Tensor


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def load_passages(passage_path):
    if passage_path.endswith('.json'):
        with open(passage_path, 'r', encoding='utf-8') as f:
            passages = json.load(f)
    elif passage_path.endswith('.jsonl'):
        with open(passage_path, 'r', encoding='utf-8') as f:
            passages = [json.loads(line.strip()) for line in f]
    return passages


def embed_passages(args, passages, model, tokenizer):
    all_ids, all_embeddings = [], []
    batch_ids, batch_texts = [], []

    with torch.no_grad():
        for i, p in enumerate(tqdm(passages, total=len(passages))):
            batch_ids.append(str(i))
            
            # need to add 'passage:' prefix for the passage
            # # NormAD
            # text = f"passage: {p['country']}\n{p['rot']}"
            # text = f"passage: {p['country']}\n{p['rot_ir']}"

            # Normad Region
            text = f"passage: {p['region']}\n{p['rot']}"
            
            # scruples
            # text = f"passage: {p['rot']}"
            # text = f"passage: {p['rot_ir']}"
            batch_texts.append(text)

            if len(batch_texts) == args.per_gpu_batch_size or i == len(passages) - 1:
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length = args.passage_max_length,
                    return_tensors="pt"
                ).to(model.device)

                outputs = model(**encoded)
                embeddings = average_pool(outputs.last_hidden_state, encoded['attention_mask'])
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1) # L2 norm
                embeddings = embeddings.cpu()

                all_ids.extend(batch_ids)
                all_embeddings.append(embeddings)

                batch_texts = []
                batch_ids = []

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return all_ids, all_embeddings


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    passages = load_passages(args.passages)
    # passages = load_dataset(args.passages, split='train')

    shard_size = len(passages) // args.num_shards
    
    start_idx = args.shard_id * shard_size
    end_idx = len(passages) if args.shard_id == args.num_shards - 1 else start_idx+shard_size

    passages = passages[start_idx:end_idx]
    print(f"Embedding {len(passages)} passages (shard {args.shard_id})")

    allids, allembeddings = embed_passages(args, passages, model, tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.prefix}.pkl")

    with open(save_path, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    print(f"Total passages processed {len(allids)}. Written to {save_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passages", type=str, required=True, help="Path to input JSONL passage file")
    parser.add_argument("--output_dir", type=str, required=True, help="dir path to save embeddings")
    parser.add_argument("--prefix", type=str, default="e5_passages", help="prefix for saved embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=128, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--passage_max_length", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", type=str, default="intfloat/e5-large-v2", help="model name or path"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")

    args = parser.parse_args()
    main(args)