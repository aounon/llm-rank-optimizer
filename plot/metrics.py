# Script to plot the mean reciprocal rank (MRR) for all catalogs
# before and after adding the STS.

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import matplotlib.patches as mpatches
from scipy import stats

from plot_dist import rank_barplot

argparser = argparse.ArgumentParser()
argparser.add_argument('input_dir', type=str, help='input directory')
args = argparser.parse_args()

input_dir = args.input_dir

ranks_df = pd.DataFrame(columns=['Catalog', 'Before', 'After','Product'])


catalog_names = {
    'coffee_machines': 'Coffee Machines',
    'cameras': 'Cameras',
    'books': 'Books',
    'election_articles': 'Political Articles',
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
                                                    'After': eval['rank_list_opt'],'Product': eval['target_product'] })], ignore_index=True)


    print(f'Rank distribution for {mode} mode:')
    # size of ranks_df per catalog
    print(ranks_df.groupby('Catalog').size())
    # Boxplot with distribution of ranks over best executions for all 10 producs
    plt.figure(figsize=(12, 8))
    distribution_ranks_df = ranks_df.copy()
    distribution_ranks_df = pd.melt(distribution_ranks_df, id_vars=['Catalog'], value_vars=['Before', 'After'], var_name='Condition', value_name='Rank')
    sns.boxplot(x='Catalog', y='Rank', hue='Condition', data=distribution_ranks_df)
    plt.ylabel('Rank', fontsize=16)
    plt.title(f'Rank Distribution Before and After Adding the STS ({mode})', fontsize=16)
    plt.legend(fontsize=16)
    plt.xlabel('')
    plt.xticks(fontsize=16)
    plt.ylim(11.5, 0.5)
    ytick_positions = range(1, 12) 
    ytick_labels = [str(i) for i in range(1, 11)] + ['not recc']
    plt.axhspan(10.5, 11.5, color='grey', alpha=0.2)    
    plt.yticks(ytick_positions, ytick_labels, fontsize=16)
    plt.legend(fontsize=16, bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, mode + '_rank_distribution.png'))


    # Boxplot with distribution of ranks over best executions for all 10 producs mean within a product
    plt.figure(figsize=(12, 8))
    distribution_ranks_df = ranks_df.copy()
    distribution_ranks_df = distribution_ranks_df.groupby(['Catalog','Product']).mean().reset_index()
    distribution_ranks_df = pd.melt(distribution_ranks_df, id_vars=['Catalog'], value_vars=['Before', 'After'], var_name='Condition', value_name='Rank')
    sns.boxplot(x='Catalog', y='Rank', hue='Condition', data=distribution_ranks_df)
    plt.ylabel('Rank', fontsize=16)
    plt.title(f'Rank Distribution Averaged per Product Before and After Adding the STS ({mode})', fontsize=16)
    plt.legend(fontsize=16)
    plt.xlabel('')
    plt.xticks(fontsize=16)
    plt.ylim(11.5, 0.5)
    ytick_positions = range(1, 12) 
    ytick_labels = [str(i) for i in range(1, 11)] + ['not recc']
    plt.axhspan(10.5, 11.5, color='grey', alpha=0.2)    
    plt.yticks(ytick_positions, ytick_labels, fontsize=16)
    plt.legend(fontsize=16, bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, mode + '_rank_distribution_mean_per_product.png'))


    ## calculate p-value for each catalog using Mann-Whitney U test (as samples are not paired)
    for catalog in catalog_names.keys():
        before = ranks_df[ranks_df['Catalog'] == catalog_names[catalog]]['Before'].dropna().to_numpy().astype(float)
        after = ranks_df[ranks_df['Catalog'] == catalog_names[catalog]]['After'].dropna().to_numpy().astype(float)
        statistic, p_value = stats.mannwhitneyu(before, after, alternative='two-sided')
        print(f'Catalog: {catalog_names[catalog]}')
        print(f'Statistic: {statistic}')
        print(f'p-value: {p_value}')
        # now common language effect size (CLES) 
        print(f'CLES: {statistic / (len(before) * len(after))}')

       
    # Mean Reciprocal Rank (MRR)
    # Create new copy of ranks_df
    for variant in ['non_appeared_zeroed', 'non_appeared_counted']:
        reciprocal_ranks_df = ranks_df.copy()
        reciprocal_ranks_df['Before'] = reciprocal_ranks_df['Before'].apply(lambda x: 1 / x if (x > 0 and x <= 10) else 0) if variant == 'non_appeared_zeroed' else 1 / reciprocal_ranks_df['Before']
        reciprocal_ranks_df['After'] = reciprocal_ranks_df['After'].apply(lambda x: 1 / x if (x > 0 and x <= 10) else 0) if variant == 'non_appeared_zeroed' else 1 / reciprocal_ranks_df['After']

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
        plt.savefig(os.path.join(input_dir, mode + '_mean_reciprocal_rank_' + variant + '.png'))

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