#!/bin/bash

num_iter=200

for run in 1 2 3
do
    for product in 6
    do
        python evaluate.py \
            --model_path "gpt-3.5-turbo" \
            --prod_idx $product \
            --sts_dir "results/transfer/product${product}/run${run}" \
            --num_iter $num_iter \
            --prod_ord random

        python plot_dist.py "results/transfer/product${product}/run${run}/eval.json"
    done
done