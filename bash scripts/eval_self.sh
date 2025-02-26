#!/bin/bash

catalog="cameras"
num_iter=200
user_msg_type="default"
model_name="vicuna"
model_path="lmsys/vicuna-7b-v1.5"

for run in {1..5}
do
    for product in {1..10}
    do
        eval_dir="results/${catalog}/self/${model_name}/${user_msg_type}/product${product}/run${run}"

        # Check if the evaluation has already been done
        if [ -f $eval_dir/done.txt ] && grep -q "done" $eval_dir/done.txt; then
            echo "Evaluation for product $product, run $run already done"
            continue
        fi

        # Evaluate the STS
        python evaluate.py \
            --model_path $model_path \
            --prod_idx $product \
            --sts_dir $eval_dir \
            --catalog $catalog \
            --num_iter $num_iter \
            --prod_ord random \
            --user_msg_type $user_msg_type      # --verbose

        # Plot the rank distribution
        python plot/plot_dist.py $eval_dir/eval.json

        # Mark the evaluation as done
        if [ ! -f $eval_dir/done.txt ]; then
            touch $eval_dir/done.txt
        fi
        echo "done" > $eval_dir/done.txt
    done
done