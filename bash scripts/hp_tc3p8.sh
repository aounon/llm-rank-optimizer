#!/bin/bash

top_candidates=3
product=8

python rank_opt.py \
    --results_dir results/hyperparameters/top_cand_${top_candidates}/product${product} \
    --catalog coffee_machines \
    --target_product_idx $product \
    --num_iter 2000 --test_iter 50 \
    --random_order --save_state \
    --top_candidates $top_candidates \
    --target_llm vicuna

echo $SLURM_NODELIST
