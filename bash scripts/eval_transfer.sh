#!/bin/bash

catalog="persons"
num_iter=10
user_msg_type="default"

for run in 3
do
    for product in 1
    do
        python evaluate.py \
            --model_path "gpt-3.5-turbo" \
            --prod_idx $product \
            --sts_dir "results/${catalog}/transfer/${user_msg_type}/product${product}/str_1/run${run}/" \
            --catalog $catalog \
            --num_iter $num_iter \
            --prod_ord random \
            --user_msg_type $user_msg_type --verbose

        python plot_dist.py "results/${catalog}/transfer/${user_msg_type}/product${product}/run${run}/eval.json"
    done
done