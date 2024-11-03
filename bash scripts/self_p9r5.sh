#!/bin/bash

product=9
run=5
catalog="election_articles"
mode="self"

python rank_opt.py \
    --results_dir results/${catalog}/${mode}/product${product}/run${run} \
    --catalog ${catalog} \
    --target_product_idx $product \
    --num_iter 2000 --test_iter 50 \
    --random_order --save_state \
    --mode ${mode}
