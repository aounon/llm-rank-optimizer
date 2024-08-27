#!/bin/bash

product=1
run=1
catalog="persons"
mode="self"
target_str_num=2

python rank_opt.py \
    --results_dir results/${catalog}/${mode}/product${product}/str_${target_str_num}/run${run} \
    --catalog ${catalog} \
    --target_product_idx $product \
    --num_iter 2000 --test_iter 50 \
    --target_str_num $target_str_num \
    --random_order \
    --mode ${mode} \
    --save_state