#!/bin/bash

catalog="coffee_machines"
num_iter=200
user_msg_type="default"

for run in 1
do
    for product in 5 1 3 4 6 8 9 10
    do
        python evaluate.py \
            --model_path "meta-llama/Llama-2-7b-chat-hf" \
            --prod_idx $product \
            --sts_dir "results/${catalog}/self/${user_msg_type}/product${product}/run${run}" \
            --catalog $catalog \
            --num_iter $num_iter \
            --prod_ord random \
            --user_msg_type $user_msg_type      # --verbose

        python plot/plot_dist.py "results/${catalog}/self/${user_msg_type}/product${product}/run${run}/eval.json"
    done
done