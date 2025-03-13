# Script to plot performance for different values of hyperparameters

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('input_dir', type=str, help='input directory')
args = argparser.parse_args()

input_dir = args.input_dir
hp_name = 'top_cand'

ranks_df = pd.DataFrame(columns=['hp_val', 'Before', 'After'])

# Get list of all directories starting with hp_name
dirs = [d for d in os.listdir(input_dir) if d.startswith(hp_name)]

for d in dirs:
    # Extract hyperparameter value
    hp_val = int(d.split('_')[-1])
    # print(f'Processing {d} with {hp_name} = {hp_val}')

    # List all eval files in the directory
    eval_files = []
    for root, _, files in os.walk(os.path.join(input_dir, d)):
        for file in files:
            if file == 'eval.json':
                eval_files.append(os.path.join(root, file))

    # Read eval files and store ranks
    for f in eval_files:
        with open(f, 'r') as file:
            eval_results = json.load(file)

        ranks_df = pd.concat([ranks_df, pd.DataFrame({'hp_val': hp_val,
                                                      'Before': eval_results['rank_list'],
                                                      'After': eval_results['rank_list_opt']})], ignore_index=True)

# Compute reciprocal ranks
ranks_df['Before'] = ranks_df['Before'].apply(lambda x: 1 / x if (x > 0 and x <= 10) else 0)
ranks_df['After'] = ranks_df['After'].apply(lambda x: 1 / x if (x > 0 and x <= 10) else 0)
ranks_df = ranks_df.melt(id_vars='hp_val', value_vars=['Before', 'After'], var_name='Condition', value_name='MRR')

# Plot the before and after reciprocal ranks for each hyperparameter value using the mean estimator
plt.figure(figsize=(10, 6))
sns.barplot(x='hp_val', y='MRR', hue='Condition', data=ranks_df)
plt.xlabel(hp_name, fontsize=16)
plt.ylabel('MRR', fontsize=16)
plt.title(f'MRR vs {hp_name}', fontsize=20)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(f'{input_dir}/mrr.png')
plt.close()
