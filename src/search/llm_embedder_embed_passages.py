import os
import argparse
import logging
import pickle
import json

import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_passages(passage_path):
    if passage_path.endswith('.json'):
        with open(passage_path, 'r', encoding='utf-8') as f:
            passages = json.load(f)
    elif passage_path.endswith('.jsonl'):
        with open(passage_path, 'r', encoding='utf-8') as f:
            passages = [json.loads(line.strip()) for line in f]
    return passages


def embed_passages(args, passages, model, tokenizer, instruction):
    all_ids, all_embeddings = [], []
    batch_ids, batch_texts = [], []

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        for i, p in enumerate(tqdm(passages, total=len(passages))):
            batch_ids.append(str(i))

            # Normad
            # text = f"{p['country']}\n{p['rot']}"
            # text = f"{p['country']}\n{p['rot_ir']}"

            # Normad Region
            text = f"{p['region']}\n{p['rot']}"

            # Scruples
            # text = f"{p['rot']}"
            # text = f"{p['rot_ir']}"

            # add instruction
            text = f"{instruction}{text}"
            batch_texts.append(text)

            if len(batch_texts) == args.per_gpu_batch_size or i == len(passages) - 1:
                doc_tokenized = tokenizer(
                    batch_texts,
                    padding=True,
                    return_tensors='pt'
                ).to(model.device)

                with torch.no_grad():
                    doc_embeddings = model(**doc_tokenized)
                    
                    # CLS pooling
                    doc_embeddings = doc_embeddings.last_hidden_state[:, 0]

                    # Normalize
                    doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
                    
                    # move to CPU
                    doc_embeddings = doc_embeddings.cpu()

                all_ids.extend(batch_ids)
                all_embeddings.append(doc_embeddings)

                batch_texts = []
                batch_ids = []

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return all_ids, all_embeddings


def main(args):
    # setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to(device)
    instruction = "Represent this document for retrieval: "

    # 모델 최적화
    model.eval()
    if not args.no_fp16 and device.type == 'cuda':
        model = model.half()
        print("Using FP16 for faster inference")

    passages = load_passages(args.passages)
    # passages = load_dataset(args.passages, split='train')

    shard_size = len(passages) // args.num_shards
    
    start_idx = args.shard_id * shard_size
    end_idx = len(passages) if args.shard_id == args.num_shards - 1 else start_idx+shard_size

    passages = passages[start_idx:end_idx]
    print(f"Embedding {len(passages)} passages (shard {args.shard_id})")
    print(f"Batch size: {args.per_gpu_batch_size}")

    allids, allembeddings = embed_passages(args, passages, model, tokenizer, instruction)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.prefix}.pkl")

    with open(save_path, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    print(f"Total passages processed {len(allids)}. Written to {save_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passages", type=str, required=True, help="Path to input JSONL passage file")
    parser.add_argument("--output_dir", type=str, required=True, help="dir path to save embeddings")
    parser.add_argument("--prefix", type=str, default="llm_embedder_passages", help="prefix for saved embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--passage_max_length", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", type=str, default="BAAI/llm-embedder", help="model name or path"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")

    args = parser.parse_args()
    main(args)