#!/bin/bash

num_iter=200

for run in 1 2 3
do
    for product in 1 2 4 6 8
    do
        python evaluate.py \
            --model_path "meta-llama/Llama-2-7b-chat-hf" \
            --prod_idx $product \
            --sts_dir "results/self/product${product}/run${run}" \
            --num_iter $num_iter \
            --prod_ord random

        python plot_dist.py "results/self/product${product}/run${run}/eval.json"
    done
done