#!/bin/bash

product=6
run=5
catalog="coffee_machines"
mode="self"
user_msg_type="default"
target_llm="vicuna"

python rank_opt.py \
    --results_dir results/${catalog}/${mode}/${target_llm}/${user_msg_type}/product${product}/run${run} \
    --catalog ${catalog} \
    --user_msg_type ${user_msg_type} \
    --target_product_idx $product \
    --num_iter 2000 --test_iter 50 \
    --random_order --save_state \
    --mode ${mode} --target_llm ${target_llm}

echo $SLURM_NODELIST
