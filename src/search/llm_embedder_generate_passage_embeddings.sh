#!/bin/bash

python3 src/search/llm_embedder_embed_passages.py \
  --passages data/normad/normad_region_norms.json \
  --output_dir index/normad_region/dense  \
  --prefix llm_embedder_normad_region \
  --model_name_or_path BAAI/llm-embedder \
  --per_gpu_batch_size 512 \
  --passage_max_length 512 \
  --shard_id 0 \
  --num_shards 1