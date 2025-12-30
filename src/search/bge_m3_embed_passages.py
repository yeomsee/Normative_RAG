# Copyright (c) Facebook, Inc. and its affiliates.
# Modified for BGE-M3 usage.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pickle
import json
import torch
import numpy as np

from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

def load_passages(passage_path):
    if not os.path.exists(passage_path):
        raise FileNotFoundError(f"Passage file not found at {passage_path}")
        
    passages = []
    if passage_path.endswith('.json'):
        with open(passage_path, 'r', encoding='utf-8') as f:
            passages = json.load(f)
    elif passage_path.endswith('.jsonl'):
        with open(passage_path, 'r', encoding='utf-8') as f:
            for line in f:
                passages.append(json.loads(line.strip()))
    else:
        raise ValueError("Unsupported file format. Please use .json or .jsonl")
    
    return passages


def embed_passages(args, passages, model):
    all_ids, all_embeddings = [], []
    batch_ids, batch_texts = [], []

    with torch.no_grad():
        for i, p in enumerate(tqdm(passages, desc="Embedding Passages")):            
            batch_ids.append(str(i + args.shard_id * (len(passages) // args.num_shards if args.num_shards > 0 else len(passages))))
            
            # NormAD
            # text = f"{p['country']}\n{p['rot']}"
            # text = f"{p['country']}\n{p['rot_ir']}"

            # Normad Region
            text = f"{p['region']}\n{p['rot']}"

            # # # Scruples
            # text = f"{p['rot']}"
            # text = f"{p['rot_ir']}"
            batch_texts.append(text)

            if len(batch_texts) == args.per_gpu_batch_size or i == len(passages) - 1:
                embeddings = model.encode(
                    batch_texts,
                    batch_size=args.per_gpu_batch_size,
                    max_length=args.passage_max_length
                )['dense_vecs']
                # embeddings = F.normalize(embeddings, p=2, dim=1)
                if isinstance(embeddings, np.ndarray):
                    embeddings = torch.from_numpy(embeddings)
                else:
                    embeddings = embeddings.detach().cpu()

                all_ids.extend(batch_ids)
                all_embeddings.append(embeddings)

                batch_texts, batch_ids = [], []

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return all_ids, all_embeddings


def main(args):
    print(f"Loading model: {args.model_name_or_path}")
    model = BGEM3FlagModel(args.model_name_or_path, use_fp16=True)

    passages = load_passages(args.passages)

    # Sharding logic
    if args.num_shards > 1:
        shard_size = len(passages) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = len(passages) if args.shard_id == args.num_shards - 1 else start_idx + shard_size
        passages_shard = passages[start_idx:end_idx]
    else:
        passages_shard = passages

    print(f"Embedding {len(passages_shard)} passages (shard {args.shard_id}/{args.num_shards})")

    all_ids, all_embeddings = embed_passages(args, passages_shard, model)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.prefix}.pkl")

    with open(save_path, mode="wb") as f:
        pickle.dump((all_ids, all_embeddings), f)

    print(f"Total passages processed: {len(all_ids)}. Written to {save_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", type=str, required=True, help="Path to input JSONL or JSON passage file. Each object must have a 'text' key.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory path to save embeddings")
    parser.add_argument("--prefix", type=str, default="bge_passages", help="Prefix for saved embedding files")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--per_gpu_batch_size", type=int, default=128, help="Batch size for the passage encoder forward pass")
    parser.add_argument("--passage_max_length", type=int, default=8192, help="Maximum number of tokens in a passage. BGE-M3 supports long contexts.")
    parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-m3", help="Model name or path")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 inference")
    args = parser.parse_args()
    main(args)