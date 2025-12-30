#!/bin/bash

python3 src/search/bge_large_en_embed_passages.py \
  --passages data/normad/normad_region_norms.json \
  --output_dir index/normad_region/dense  \
  --prefix bge_large_en_normad_region \
  --model_name_or_path BAAI/bge-large-en-v1.5 \
  --per_gpu_batch_size 512 \
  --passage_max_length 512