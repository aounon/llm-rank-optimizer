import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import pandas as pd
import os
import numpy as np
import argparse

# Set seaborn style
# sns.set(style="darkgrid")
sns.set_style("darkgrid")

def create_rank_list(rank_dist, num_prod, total_values=20):
    # Create a list of ranks of length total_values preserving the distribution in rank_dist
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

def rank_barplot(rank_list, rank_list_opt, num_prod, plot_title="Rank Distribution", axes=None, title_fontsize=16, plot_fontsize=14):
    rank_df = pd.DataFrame({
        "Rank": range(1, num_prod + 2),
        "Before": [rank_list.count(i) for i in range(1, num_prod + 2)],
        "After": [rank_list_opt.count(i) for i in range(1, num_prod + 2)]
    })

    # Convert frequency to fraction
    total_values_before = rank_df["Before"].sum()
    total_values_after = rank_df["After"].sum()
    rank_df["Before"] = (rank_df["Before"] / total_values_before)
    rank_df["After"] = (rank_df["After"] / total_values_after)

    rank_df = rank_df.melt("Rank", var_name="Rank Type", value_name="Frequency")

    # Calculate confidence intervals
    num_rows = rank_df.shape[0]
    for i in range(num_rows):
        row = rank_df.iloc[i]
        freq = row["Frequency"]
        error = 1.96 * np.sqrt(freq * (1 - freq) / total_values_before)
        lower_bound = max(0, freq - error)
        upper_bound = min(1, freq + error)
        rank_df = pd.concat([rank_df, pd.DataFrame({"Rank": [row["Rank"]], "Rank Type": [row["Rank Type"]], "Frequency": [lower_bound]})])
        rank_df = pd.concat([rank_df, pd.DataFrame({"Rank": [row["Rank"]], "Rank Type": [row["Rank Type"]], "Frequency": [upper_bound]})])

    # Scale up to 100
    rank_df["Frequency"] *= 100

    if axes is None:
        axes = plt.gca()

    sns.barplot(x="Rank", y="Frequency", hue="Rank Type", data=rank_df, estimator='median', errorbar="ci", ax=axes)
    y_min, y_max = axes.get_ylim()
    axes.fill_between([9.5, 10.5], y_min, y_max, color="grey", alpha=0.3, zorder=0)
    axes.set_title(plot_title, fontsize=title_fontsize)
    axes.set_xlabel("Rank", fontsize=title_fontsize)
    axes.set_ylabel("Percentage", fontsize=title_fontsize)
    axes.set_xticks(range(0, num_prod + 1))
    axes.set_xticklabels([str(i) if i <= num_prod else 'NR' for i in range(1, num_prod + 2)], fontsize=14)
    axes.tick_params(axis='x', labelsize=plot_fontsize)
    axes.tick_params(axis='y', labelsize=plot_fontsize)
    axes.set_ylim(y_min, y_max)
    grey_patch = mpatches.Patch(color='grey', alpha=0.3, label='Not Recommended')
    axes.legend(handles=axes.get_legend_handles_labels()[0] + [grey_patch], fontsize=plot_fontsize)
    plt.tight_layout()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot the results of the evaluation")
    parser.add_argument("results_file", type=str, help="The file containing the evaluation results")
    args = parser.parse_args()

    results_file = args.results_file
    # Get the directory of the results file
    res_file_dir = os.path.dirname(results_file)
    # print(f"Results file: {results_file}")
    # print(f"Results directory: {res_file_dir}")

    # Load json file
    with open(results_file, "r") as file:
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
    rank_list_approx = create_rank_list(rank_dist, num_prod)
    rank_list_opt_approx = create_rank_list(rank_dist_opt, num_prod)
    rank_list_cleaned_approx = create_rank_list(rank_dist_cleaned, num_prod)
    rank_list_opt_cleaned_approx = create_rank_list(rank_dist_opt_cleaned, num_prod)

    plot_ranks(rank_list_approx, rank_list_opt_approx, num_prod, res_file_dir + "/ranks.png", plot_title=f"Rank Distribution for\n{target_product}")
    plot_ranks(rank_list_cleaned_approx, rank_list_opt_cleaned_approx, num_prod, res_file_dir + "/ranks_cleaned.png", plot_title=f"Rank Distribution for\n{target_product} (Rec Only)")

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    rank_barplot(data["rank_list"], data["rank_list_opt"], num_prod, plot_title=f"Rank Distribution for {target_product}", axes=axes)
    fig.savefig(os.path.join(res_file_dir, "rank_barplot.png"))

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    rank_barplot(data["rank_list_cleaned"], data["rank_list_opt_cleaned"], num_prod, plot_title=f"Rank Distribution for {target_product} (Rec Only)", axes=axes)
    fig.savefig(os.path.join(res_file_dir, "rank_barplot_cleaned.png"))

    # Plot advantage
    plot_advantage(advantage, res_file_dir + "/advantage.png", plot_title=f"Rank Advantage for {target_product}")
    plot_advantage(advantage_cleaned, res_file_dir + "/advantage_cleaned.png", plot_title=f"Rank Advantage for {target_product} (Rec Only)")
