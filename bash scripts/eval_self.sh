#!/bin/bash

catalog="books"
num_iter=200

for run in 1
do
    for product in {1..10}
    do
        python evaluate.py \
            --model_path "meta-llama/Llama-2-7b-chat-hf" \
            --prod_idx $product \
            --sts_dir "results/${catalog}/self/product${product}/run${run}" \
            --catalog $catalog \
            --num_iter $num_iter \
            --prod_ord random

        python plot_dist.py "results/${catalog}/self/product${product}/run${run}/eval.json"
    done
done