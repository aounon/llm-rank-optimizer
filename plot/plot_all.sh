#!/bin/bash

catalogs=("coffee_machines" "cameras" "books" "election_articles")

modes=("self" "transfer")

for catalog in "${catalogs[@]}"; do
    for mode in "${modes[@]}"; do
        
        path="results/$catalog/$mode/default"
        
        if [ -d "$path" ]; then
            echo "Running: python3 plot/aggregate_advantage.py \"$path\""
            python3 plot/aggregate_advantage.py "$path"
        else
            echo "Warning: Directory $path does not exist, skipping..."
        fi
    done
done

python3 plot/metrics.py results