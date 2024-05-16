#!/bin/bash

product=4
run=2
python rank_opt.py \
    --results_dir results/transfer/product${product}/run${run} \
    --target_product_idx $product \
    --num_iter 2000 --test_iter 50 \
    --random_order \
    --mode transfer
