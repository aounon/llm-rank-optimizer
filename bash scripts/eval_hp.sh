#!/bin/bash

num_iter=200
model_path="lmsys/vicuna-7b-v1.5"

for top_candidates in 20 25 30
do
    for product in {1..10}
    do
        eval_dir="results/hyperparameters/top_cand_${top_candidates}/product${product}"

        # Check if the evaluation has already been done
        if [ -f $eval_dir/done.txt ] && grep -q "done" $eval_dir/done.txt; then
            echo "Evaluation for top_candidates=${top_candidates} and product=${product} already done. Skipping..."
            continue
        fi

        # Evaluate the STS
        python evaluate.py \
            --model_path $model_path \
            --prod_idx $product \
            --sts_dir $eval_dir \
            --catalog coffee_machines \
            --num_iter $num_iter \
            --prod_ord random       # --verbose

        # Plot the rank distribution
        python plot/plot_dist.py $eval_dir/eval.json

        # Mark the evaluation as done
        if [ ! -f $eval_dir/done.txt ]; then
            touch $eval_dir/done.txt
        fi
        echo "done" > $eval_dir/done.txt
    done
done

echo $SLURM_NODELIST