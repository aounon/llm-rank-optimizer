import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import pandas as pd
# import numpy as np
import argparse

# Set seaborn style
# sns.set(style="darkgrid")
sns.set_style("darkgrid")

def create_rank_list(rank_dist, num_prod, total_values=20):
    rank_list = []
    cumulative_error = 0.0

    # Calculate the total sum of percentages
    total_percentage = sum(rank_dist.values())

    for rank, percentage in rank_dist.items():
        # Calculate the exact count and adjust for cumulative error
        exact_count = (percentage / total_percentage) * total_values + cumulative_error
        count = round(exact_count)

        # Update the cumulative error
        cumulative_error += (exact_count - count)

        # Add the rank to the list
        rank_list.extend([int(rank)] * count)

    # Adjust the list length to exactly 25 if necessary
    if len(rank_list) > total_values:
        rank_list = rank_list[:total_values]
    elif len(rank_list) < total_values:
        rank_list.extend([num_prod + 1] * (total_values - len(rank_list)))

    return rank_list

def plot_ranks(rank_list, rank_list_opt, num_prod, save_path, plot_title="Product Rank Distribution"):
    ranks_df = pd.DataFrame({
        "Before": rank_list,
        "After": rank_list_opt
    })
    plt.figure(figsize=(5, 6))
    sns.swarmplot(data=ranks_df, size=7)
    plt.fill_between([-1, 2], (num_prod+1) * 1.04, num_prod + 0.5, color="grey", alpha=0.3, zorder=0)
    plt.title(plot_title, fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(-0.5, 1.5)
    plt.ylabel("Rank", fontsize=16)
    plt.ylim((num_prod+1) * 1.04, 1 - ((num_prod+1) * 0.04))
    plt.yticks(range(num_prod, 0, -1), fontsize=14)
    grey_patch = mpatches.Patch(color='grey', alpha=0.3, label='Not Recommended')
    plt.legend(handles=[grey_patch])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_advantage(advantage, save_path, plot_title="Rank Advantage"):
    plt.figure(figsize=(7, 5))
    plt.bar(["Advantage"], [advantage['1']], color="tab:green")
    plt.bar(["No Advantage"], [advantage['0']], color="tab:blue")
    plt.bar(["Disadvantage"], [advantage['-1']], color="tab:brown")
    plt.title(plot_title, fontsize=16)
    plt.ylabel("Percentage", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 103)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot the results of the evaluation")
    parser.add_argument("--prod_tag", type=str, help="Tag of the product to rank", default='3_of_ef')
    args = parser.parse_args()

    prod_tag = args.prod_tag

    # Load json file
    file_path = f'results/eval_prod_{prod_tag}.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    target_product = data["target_product"]
    rank_dist = data["rank_dist"]
    rank_dist_opt = data["rank_dist_opt"]
    rank_dist_cleaned = data["rank_dist_cleaned"]
    rank_dist_opt_cleaned = data["rank_dist_opt_cleaned"]
    advantage = data["advantage"]
    advantage_cleaned = data["advantage_cleaned"]

    num_prod = 10

    # Create rank lists
    rank_list = create_rank_list(rank_dist, num_prod)
    rank_list_opt = create_rank_list(rank_dist_opt, num_prod)
    rank_list_cleaned = create_rank_list(rank_dist_cleaned, num_prod)
    rank_list_opt_cleaned = create_rank_list(rank_dist_opt_cleaned, num_prod)

    plot_ranks(rank_list, rank_list_opt, num_prod, file_path.replace(".json", "_ranks.png"), plot_title=f"Rank Distribution for\n{target_product}")
    plot_ranks(rank_list_cleaned, rank_list_opt_cleaned, num_prod, file_path.replace(".json", "_ranks_cleaned.png"), plot_title=f"Rank Distribution for\n{target_product} (Rec Only)")

    # Plot advantage
    plot_advantage(advantage, file_path.replace(".json", "_advantage.png"), plot_title=f"Rank Advantage for {target_product}")
    plot_advantage(advantage_cleaned, file_path.replace(".json", "_advantage_cleaned.png"), plot_title=f"Rank Advantage for {target_product} (Rec Only)")
