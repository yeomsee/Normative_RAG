#!/bin/bash

python3 src/search/contriever_embed_passages.py \
  --passages data/scruples/3shot/norms.json \
  --output_dir data/index/scruples/dense/3shot  \
  --prefix contriever_scruples \
  --model_name_or_path facebook/contriever \
  --per_gpu_batch_size 512 \
  --passage_max_length 512 \
  --shard_id 0 \
  --num_shards 1