# Script to plot the mean reciprocal rank (MRR) for all catalogs
# before and after adding the STS.

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

from plot_dist import rank_barplot

argparser = argparse.ArgumentParser()
argparser.add_argument('input_dir', type=str, help='input directory')
args = argparser.parse_args()

input_dir = args.input_dir

ranks_df = pd.DataFrame(columns=['Catalog', 'Before', 'After'])

catalog_names = {
    'coffee_machines': 'Coffee Machines',
    'cameras': 'Cameras',
    'books': 'Books'
}

for mode in ['self', 'transfer']:
    for catalog in catalog_names.keys():
        path_to_products = os.path.join(input_dir, catalog, mode, 'default')

        # List all product dirtecories
        dirs = [d for d in os.listdir(path_to_products) if os.path.isdir(os.path.join(path_to_products, d))]
        dirs.sort()

        for dir in dirs:
            # List all run directories in the product directory
            product_dir = os.path.join(path_to_products, dir)
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
                    ranks_df = pd.concat([ranks_df, pd.DataFrame({'Catalog': catalog_names[catalog], 'Before': eval['rank_list'],
                                                    'After': eval['rank_list_opt']})], ignore_index=True)

    # Mean Reciprocal Rank (MRR)
    # Create new copy of ranks_df
    reciprocal_ranks_df = ranks_df.copy()
    reciprocal_ranks_df['Before'] = reciprocal_ranks_df['Before'].apply(lambda x: 1 / x if (x > 0 and x <= 10) else 0)
    reciprocal_ranks_df['After'] = reciprocal_ranks_df['After'].apply(lambda x: 1 / x if (x > 0 and x <= 10) else 0)

    # Plot the mean reciprocal rank (MRR) for all catalogs before and after adding the STS
    reciprocal_ranks_df = pd.melt(reciprocal_ranks_df, id_vars=['Catalog'], value_vars=['Before', 'After'], 
                                var_name='Condition', value_name='MRR')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Catalog', y='MRR', hue='Condition', data=reciprocal_ranks_df)
    plt.ylabel('Mean Reciprocal Rank (MRR)', fontsize=16)
    plt.title('Mean Reciprocal Rank (MRR) Before and After Adding the STS', fontsize=16)
    plt.legend(fontsize=16)
    plt.xlabel('')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(input_dir, mode + '_mrr.png'))

    # Top 3 Rank
    # Create new copy of ranks_df
    top_3_df = ranks_df.copy()
    top_3_df['Before'] = top_3_df['Before'].apply(lambda x: 1 if x <= 3 else 0)
    top_3_df['After'] = top_3_df['After'].apply(lambda x: 1 if x <= 3 else 0)

    # Scale up to 100
    top_3_df['Before'] *= 100
    top_3_df['After'] *= 100

    top_3_df = pd.melt(top_3_df, id_vars=['Catalog'], value_vars=['Before', 'After'],
                    var_name='Condition', value_name='Top 3')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Catalog', y='Top 3', hue='Condition', data=top_3_df)
    plt.ylabel('Percentage of Being in Top 3 Rank', fontsize=16)
    plt.title('Percentage of Being in Top 3 Rank Before and After Adding the STS', fontsize=16)
    plt.legend(fontsize=16)
    plt.xlabel('')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(input_dir, mode + '_top3.png'))

    # Median Rank
    median_ranks_df = ranks_df.copy()
    median_ranks_df = pd.melt(median_ranks_df, id_vars=['Catalog'], value_vars=['Before', 'After'],
                              var_name='Condition', value_name='Rank')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Catalog', y='Rank', hue='Condition', data=median_ranks_df, estimator='median')
    plt.ylabel('Median Rank', fontsize=16)
    plt.title('Median Rank Before and After Adding the STS', fontsize=16)
    plt.legend(fontsize=16)
    plt.xlabel('')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(input_dir, mode + '_median_rank.png'))

    # Plot the rank distribution for all catalogs before and after adding the STS
    rank_barplot(ranks_df['Before'].tolist(), ranks_df['After'].tolist(), 10, os.path.join(input_dir, mode + '_rank_dist.png'))

    # Plot advantage for all catalogs
    advantage_df = pd.DataFrame(columns=['Catalog', 'Advantage', 'No Advantage', 'Disadvantage', 'Net Advantage'])

    for catalog in catalog_names.keys():
        advantage = {'Advantage': 0, 'No Advantage': 0, 'Disadvantage': 0, 'Total': 0}
        for index, row in ranks_df.iterrows():
            if row['Catalog'] == catalog_names[catalog]:
                if row['Before'] > row['After']:
                    advantage['Advantage'] += 1
                elif row['Before'] == row['After']:
                    advantage['No Advantage'] += 1
                else:
                    advantage['Disadvantage'] += 1
                advantage['Total'] += 1

        advantage_percentage = advantage['Advantage'] / advantage['Total']
        no_advantage_percentage = advantage['No Advantage'] / advantage['Total']
        disadvantage_percentage = advantage['Disadvantage'] / advantage['Total']

        # Calculate 95% confidence interval
        advantage_error = 1.96 * np.sqrt(advantage_percentage * (1 - advantage_percentage) / advantage['Total'])
        no_advantage_error = 1.96 * np.sqrt(no_advantage_percentage * (1 - no_advantage_percentage) / advantage['Total'])
        disadvantage_error = 1.96 * np.sqrt(disadvantage_percentage * (1 - disadvantage_percentage) / advantage['Total'])

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

        advantage_df = pd.concat([advantage_df, pd.DataFrame({'Catalog': catalog_names[catalog], 'Advantage': advantage_percentage,
                                                            'No Advantage': no_advantage_percentage, 'Disadvantage': disadvantage_percentage,
                                                            'Net Advantage': net_advantage_percentage}, index=[0])], ignore_index=True)
        advantage_df = pd.concat([advantage_df, pd.DataFrame({'Catalog': catalog_names[catalog], 'Advantage': advantage_percentage - advantage_error,
                                                            'No Advantage': no_advantage_percentage - no_advantage_error, 'Disadvantage': disadvantage_percentage - disadvantage_error,
                                                            'Net Advantage': net_advantage_percentage - net_advantage_error}, index=[0])], ignore_index=True)
        advantage_df = pd.concat([advantage_df, pd.DataFrame({'Catalog': catalog_names[catalog], 'Advantage': advantage_percentage + advantage_error,
                                                            'No Advantage': no_advantage_percentage + no_advantage_error, 'Disadvantage': disadvantage_percentage + disadvantage_error,
                                                            'Net Advantage': net_advantage_percentage + net_advantage_error}, index=[0])], ignore_index=True)
        
    advantage_df = pd.melt(advantage_df, id_vars=['Catalog'], value_vars=['Advantage', 'Disadvantage', 'Net Advantage'],
                            var_name='Condition', value_name='Percentage')
    # advantage_df = pd.melt(advantage_df, id_vars=['Catalog'], value_vars=['Advantage', 'No Advantage', 'Disadvantage', 'Net Advantage'],
    #                         var_name='Condition', value_name='Percentage')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Catalog', y='Percentage', hue='Condition', data=advantage_df, estimator='median', errorbar='ci')
    plt.ylabel('Percentage', fontsize=16)
    plt.title('Advantage for all Catalogs', fontsize=16)
    plt.legend(fontsize=16)
    plt.xlabel('')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(input_dir, mode + '_advantage.png'))