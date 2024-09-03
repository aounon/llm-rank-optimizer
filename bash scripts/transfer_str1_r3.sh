#!/bin/bash

product=1
run=3
catalog="persons"
mode="transfer"
user_msg_type="default"
target_str_num=1

python rank_opt.py \
    --results_dir results/${catalog}/${mode}/${user_msg_type}/product${product}/str_${target_str_num}/run${run} \
    --catalog ${catalog} \
    --user_msg_type ${user_msg_type} \
    --target_product_idx $product \
    --num_iter 2000 --test_iter 50 \
    --target_str_num $target_str_num \
    --random_order \
    --mode ${mode} \
    --save_state
