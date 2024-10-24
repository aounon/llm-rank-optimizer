# Plot the aggregated advantage over all products, picking the best run for each product
import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('input_dir', type=str, help='input directory')
args = argparser.parse_args()

input_dir = args.input_dir
result_dict = {
    'Advantage': 0,
    'No Advantage': 0,
    'Disadvantage': 0,
    'Total': 0
}

# List all directories in the input directory
dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
dirs.sort()

for dir in dirs:
    # List all run directories in the product directory
    product_dir = os.path.join(input_dir, dir)
    run_dirs = [d for d in os.listdir(product_dir) if os.path.isdir(os.path.join(product_dir, d))]
    best_run_dir = None
    best_run_advantage = None
    for run_dir in run_dirs:
        run_dir_path = os.path.join(product_dir, run_dir)
        if os.path.exists(os.path.join(run_dir_path, 'eval.json')):
            with open(os.path.join(run_dir_path, 'eval.json'), 'r') as f:
                eval = json.load(f)
                if best_run_advantage is None or (eval['advantage']['1'] - eval['advantage']['-1'] > best_run_advantage):
                    best_run_dir = run_dir
                    best_run_advantage = eval['advantage']['1'] - eval['advantage']['-1']
    
    if best_run_dir is not None:
        with open(os.path.join(product_dir, best_run_dir, 'eval.json'), 'r') as f:
            eval = json.load(f)
            for i in range(len(eval['rank_list'])):
                if eval['rank_list'][i] > eval['rank_list_opt'][i]:
                    result_dict['Advantage'] += 1
                elif eval['rank_list'][i] == eval['rank_list_opt'][i]:
                    result_dict['No Advantage'] += 1
                else:
                    result_dict['Disadvantage'] += 1
                result_dict['Total'] += 1

print(result_dict)

advantage_percentage = result_dict['Advantage'] / result_dict['Total']
no_advantage_percentage = result_dict['No Advantage'] / result_dict['Total']
disadvantage_percentage = result_dict['Disadvantage'] / result_dict['Total']

# Calculate 95% confidence interval
advantage_error = 1.96 * np.sqrt(advantage_percentage * (1 - advantage_percentage) / result_dict['Total'])
no_advantage_error = 1.96 * np.sqrt(no_advantage_percentage * (1 - no_advantage_percentage) / result_dict['Total'])
disadvantage_error = 1.96 * np.sqrt(disadvantage_percentage * (1 - disadvantage_percentage) / result_dict['Total'])

# Scale up to 100
advantage_percentage *= 100
no_advantage_percentage *= 100
disadvantage_percentage *= 100
advantage_error *= 100
no_advantage_error *= 100
disadvantage_error *= 100

# Calculate net advantage percentage and error
net_advantage_percentage = advantage_percentage - disadvantage_percentage
net_advantage_error = np.sqrt(advantage_error ** 2 + disadvantage_error ** 2)

fig, ax = plt.subplots()
ax.bar(['Advantage'], [advantage_percentage], yerr=[advantage_error], capsize=5, label='Advantage', color='tab:blue')
ax.bar(['Disadvantage'], [disadvantage_percentage], yerr=[disadvantage_error], capsize=5, label='Disadvantage', color='tab:brown')
ax.bar(['Net Advantage'], [net_advantage_percentage], yerr=[net_advantage_error], capsize=5, label='Net Advantage', color='tab:green')
# ax.bar(['No Advantage'], [no_advantage_percentage], yerr=[no_advantage_error], capsize=5, label='No Advantage', color='tab:purple')
ax.set_ylabel('Percentage', fontsize=16)
# ax.set_ylim(0, 100)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(os.path.join(input_dir, 'aggregated_advantage.png'))