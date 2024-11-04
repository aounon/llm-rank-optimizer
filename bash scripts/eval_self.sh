#!/bin/bash

catalog="coffee_machines"
num_iter=200
user_msg_type="default"

for run in {1..5}
do
    for product in {1..10}
    do
        eval_dir="results/${catalog}/self/${user_msg_type}/product${product}/run${run}"

        # Check if the evaluation has already been done
        if [ -f $eval_dir/done.txt ] && grep -q "done" $eval_dir/done.txt; then
            echo "Evaluation for product $product already done"
            continue
        fi

        # Evaluate the STS
        python evaluate.py \
            --model_path "meta-llama/Llama-2-7b-chat-hf" \
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