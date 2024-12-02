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
argparser.add_argument('--num_products', type=int, default=10, help='number of products to use for plots')
args = argparser.parse_args()

input_dir = args.input_dir
num_products = args.num_products

catalog_names = {
    'coffee_machines': 'Coffee Machines',
    'cameras': 'Cameras',
    # 'books': 'Books',
    'election_articles': 'Political Articles',
}

for mode in ['self', 'transfer']:
    ranks_df = pd.DataFrame(columns=['Catalog', 'Before', 'After','Product'])
    for catalog in catalog_names.keys():

        path_to_products = os.path.join(input_dir, catalog, mode, 'default')

        # List all product dirtecories
        dirs = [d for d in os.listdir(path_to_products) if os.path.isdir(os.path.join(path_to_products, d))]
        dirs.sort()

        best_run_per_product = []

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
            best_run_per_product.append((product_dir, best_run_dir, best_run_advantage))

            # if best_run_dir is not None:
            #     with open(os.path.join(product_dir, best_run_dir, 'eval.json'), 'r') as f:
            #         eval = json.load(f)
            #         ranks_df = pd.concat([ranks_df, pd.DataFrame({'Catalog': catalog_names[catalog], 'Before': eval['rank_list'],
            #                                         'After': eval['rank_list_opt'],'Product': eval['target_product']})], ignore_index=True)
                    
        # Sort by advantage in descending order keeping None at the end
        best_run_per_product.sort(key=lambda x: x[2] if x[2] is not None else -np.inf, reverse=True)

        for product_dir, best_run_dir, best_run_advantage in best_run_per_product[:num_products]:
            if best_run_dir is not None:
                with open(os.path.join(product_dir, best_run_dir, 'eval.json'), 'r') as f:
                    eval = json.load(f)
                    ranks_df = pd.concat([ranks_df, pd.DataFrame({'Catalog': catalog_names[catalog], 'Before': eval['rank_list'],
                                                    'After': eval['rank_list_opt'],'Product': eval['target_product']})], ignore_index=True)
                    # if len(eval['rank_list']) < 200:
                    #     print(product_dir, best_run_dir, eval['target_product'], len(eval['rank_list']))

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
    ytick_labels = [str(i) for i in range(1, 11)] + ['NR']
    plt.axhspan(10.5, 11.5, color='grey', alpha=0.3, zorder=0)
    plt.yticks(ytick_positions, ytick_labels, fontsize=16)
    grey_patch = mpatches.Patch(color='grey', alpha=0.3, label='Not Recommended')
    plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + [grey_patch], fontsize=16, bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, mode + '_p' + str(num_products) + '_rank_dist_boxplt.png'))


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
    ytick_labels = [str(i) for i in range(1, 11)] + ['NR']
    plt.axhspan(10.5, 11.5, color='grey', alpha=0.3, zorder=0)
    plt.yticks(ytick_positions, ytick_labels, fontsize=16)
    plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + [grey_patch], fontsize=16, bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, mode + '_p' + str(num_products) + '_rank_dist_mean_per_prod.png'))


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
    reciprocal_ranks_df = ranks_df.copy()
    reciprocal_ranks_df['Before'] = reciprocal_ranks_df['Before'].apply(lambda x: 1 / x if (x > 0 and x <= 10) else 0)
    reciprocal_ranks_df['After'] = reciprocal_ranks_df['After'].apply(lambda x: 1 / x if (x > 0 and x <= 10) else 0)
    reciprocal_ranks_df = pd.melt(reciprocal_ranks_df, id_vars=['Catalog'], value_vars=['Before', 'After'], 
                                var_name='Condition', value_name='MRR')

    # Rank 1 percentage
    top_k_df = ranks_df.copy()
    top_k_df['Before'] = top_k_df['Before'].apply(lambda x: 1 if x <= 1 else 0)
    top_k_df['After'] = top_k_df['After'].apply(lambda x: 1 if x <= 1 else 0)
    top_k_df['Before'] *= 100
    top_k_df['After'] *= 100
    top_k_df = pd.melt(top_k_df, id_vars=['Catalog'], value_vars=['Before', 'After'],
                    var_name='Condition', value_name='Rank 1 Percentage')

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
        advantage_error = 1.96 * np.sqrt(advantage_percentage * (1 - advantage_percentage) / advantage['Total'])
        no_advantage_error = 1.96 * np.sqrt(no_advantage_percentage * (1 - no_advantage_percentage) / advantage['Total'])
        disadvantage_error = 1.96 * np.sqrt(disadvantage_percentage * (1 - disadvantage_percentage) / advantage['Total'])
        advantage_percentage *= 100
        no_advantage_percentage *= 100
        disadvantage_percentage *= 100
        advantage_error *= 100
        no_advantage_error *= 100
        disadvantage_error *= 100
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

    # Plot MRR, Top-k, and Advantage in a single figure
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    sns.barplot(x='Catalog', y='MRR', hue='Condition', data=reciprocal_ranks_df, ax=axes[0])
    axes[0].set_ylabel('MRR', fontsize=22)
    axes[0].set_title('Mean Reciprocal Rank (MRR)', fontsize=22)
    axes[0].legend(fontsize=18)
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', labelsize=16)
    axes[0].tick_params(axis='y', labelsize=16)

    sns.barplot(x='Catalog', y='Rank 1 Percentage', hue='Condition', data=top_k_df, ax=axes[1])
    axes[1].set_ylabel(f'Percentage', fontsize=22)
    axes[1].set_title(f'Percentage of Being Rank 1', fontsize=22)
    axes[1].legend(fontsize=18)
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', labelsize=16)
    axes[1].tick_params(axis='y', labelsize=16)

    sns.barplot(x='Catalog', y='Percentage', hue='Condition', data=advantage_df, estimator='median', errorbar='ci', ax=axes[2])
    axes[2].set_ylabel('Percentage', fontsize=22)
    axes[2].set_title('Advantage for all Catalogs', fontsize=22)
    axes[2].legend(fontsize=18)
    axes[2].set_xlabel('')
    axes[2].tick_params(axis='x', labelsize=16)
    axes[2].tick_params(axis='y', labelsize=16)

    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(input_dir, mode + '_p' + str(num_products) + '_combined_plots.png'))

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
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, mode + '_p' + str(num_products) + '_median_rank.png'))

    # Plot the rank distribution for all catalogs before and after adding the STS
    rank_barplot(ranks_df['Before'].tolist(), ranks_df['After'].tolist(), 10, os.path.join(input_dir, mode + '_p' + str(num_products) + '_rank_dist.png'))

    # Plot the net advantage frequency histogram for all products in all catalogs
    net_advantages = []
    product_names = ranks_df['Product'].unique()
    for product in product_names:
        advantage = 0
        disadvantage = 0
        total = 0
        ranks_product_df = ranks_df[ranks_df['Product'] == product]
        catalog = ranks_product_df['Catalog'].iloc[0]
        for index, row in ranks_product_df.iterrows():
            if row['Before'] > row['After']:
                advantage += 1
            elif row['Before'] < row['After']:
                disadvantage += 1
            total += 1
        net_advantages.append({
            'Product': product,
            'Net Advantage': ((advantage - disadvantage) / total) * 100,
            'Catalog': catalog
        })

    # for adv in net_advantages:
    #     print(adv)
    # exit()

    net_advantage_positive = [adv['Net Advantage'] for adv in net_advantages if adv['Net Advantage'] >= 0]
    net_advantage_negative = [adv['Net Advantage'] for adv in net_advantages if adv['Net Advantage'] < 0]

    plt.figure(figsize=(10, 6))
    plt.hist(net_advantage_positive, label='Positive', range=(0, 100), bins=20, color='tab:green')
    plt.hist(net_advantage_negative, label='Negative', range=(-100, 0), bins=20, color='tab:red')
    plt.xlabel('Net Advantage (%)', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title('Net Advantage Frequency Histogram', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(-100, 100)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, mode + '_p' + str(num_products) + '_net_adv_freq.png'))

    # Plot the net advantage values in decreasing order 
    net_advantages = [adv for adv in net_advantages if adv['Catalog'] != 'Political Articles']
    net_advantages.sort(key=lambda x: x['Net Advantage'])
    plt.figure(figsize=(6, 10))
    bars = plt.barh(range(len(net_advantages)), [adv['Net Advantage'] for adv in net_advantages], color=['tab:green' if adv['Net Advantage'] >= 0 else 'tab:red' for adv in net_advantages])
    plt.yticks(range(len(net_advantages)), [''] * len(net_advantages))  # Remove y-ticks
    for bar, adv in zip(bars, net_advantages):
        plt.text(1.5, bar.get_y() + bar.get_height() / 2, adv['Product'], ha='left', va='center', fontsize=14)
    plt.ylabel('Products', fontsize=16)
    plt.xlabel('Net Advantage (%)', fontsize=16)
    plt.title('Net Advantage for all Products', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, mode + '_p' + str(num_products) + '_net_adv.png'))

    # plt.figure(figsize=(10, 6))
    # bars = plt.bar(range(len(net_advantages)), [adv['Net Advantage'] for adv in net_advantages], color=['tab:green' if adv['Net Advantage'] >= 0 else 'tab:red' for adv in net_advantages])
    # plt.xticks(range(len(net_advantages)), [''] * len(net_advantages))  # Remove x-ticks
    # for bar, adv in zip(bars, net_advantages):
    #     plt.text(0.05 + bar.get_x() + bar.get_width() / 2, 1.5, adv['Product'], ha='center', va='bottom', fontsize=14, rotation=90)
    # # plt.xticks(rotation=90)
    # plt.xlabel('Products', fontsize=16)
    # plt.ylabel('Net Advantage (%)', fontsize=16)
    # plt.title('Net Advantage for all Products', fontsize=16)
    # # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=16)
    # plt.tight_layout()
    # plt.savefig(os.path.join(input_dir, mode + '_p' + str(num_products) + '_net_adv.png'))
