import transformers
import torch
import json
import numpy as np
import argparse

from rank_opt import rank_products

def get_rank(system_prompt, user_msg, product_lines, target_product, product_names, model, tokenizer, device):
    prompt = system_prompt
    for line in product_lines:
        prompt += line

    prompt += "\n" + user_msg + " [/INST]"
    # print(f'INPUT PROMPT: {prompt}')
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    response = model.generate(input_ids, model.generation_config, max_new_tokens=800)
    # print(f'MODEL RAW OUT: {tokenizer.decode(response[0, :])}')
    model_output_new = tokenizer.decode(response[0, len(input_ids[0]):], skip_special_tokens=True)
    # print(f'MODEL OUTPUT: {model_output_new}')
    rank = rank_products(model_output_new, product_names)[target_product]
    # print(f"Rank: {rank}")
    return rank

def rank_distribution(rank_list):
    # Rank counts
    rank_dist = {}
    for value in rank_list:
        if value in rank_dist:
            rank_dist[value] += 1
        else:
            rank_dist[value] = 1

    # Turn counts into percentages rounded to 2 decimal places
    total = len(rank_list)
    for key in rank_dist:
        rank_dist[key] = round(rank_dist[key] / total * 100, 2)

    # Sort by key and return
    return dict(sorted(rank_dist.items()))

def rank_advantage(rank_list, rank_list_opt):
    rank_diff = [rank_list[i] - rank_list_opt[i] for i in range(len(rank_list))]

    # Get sign of rank difference
    advantage = {1: 0, 0: 0, -1: 0}
    for value in rank_diff:
        if value > 0:
            advantage[1] += 1
        elif value < 0:
            advantage[-1] += 1
        else:
            advantage[0] += 1

    # Turn counts into percentages rounded to 2 decimal places
    total_diff = len(rank_diff)

    if total_diff > 0:
        for key in advantage:
            advantage[key] = round(advantage[key] / total_diff * 100, 2)

    return advantage

def clean_rank_lists(rank_list, rank_list_opt, num_products):
    # Remove rows where product is not recommended in both cases
    num_ranks = len(rank_list)
    rank_list_cleaned = []
    rank_list_opt_cleaned = []
    for i in range(num_ranks):
        if rank_list[i] > num_products and rank_list_opt[i] > num_products:
            continue
        else:
            rank_list_cleaned.append(rank_list[i])
            rank_list_opt_cleaned.append(rank_list_opt[i])

    return rank_list_cleaned, rank_list_opt_cleaned

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate the performance of the model")
    parser.add_argument("--prod_idx", type=int, help="Index of the product to rank", default=3)
    parser.add_argument("--num_iter", type=int, help="Number of iterations to run", default=50)
    parser.add_argument("--opt_prod_ord", type=str, choices=["random", "fixed"], help="Order of products during optimization", default="random")
    parser.add_argument("--eval_prod_ord", type=str, choices=["random", "fixed"], help="Order of products during evaluation", default="random")
    args = parser.parse_args()

    prod_idx = args.prod_idx
    num_iter = args.num_iter
    opt_prod_ord = args.opt_prod_ord
    eval_prod_ord = args.eval_prod_ord

    save_path = f'results/eval_prod_{prod_idx}'
    if opt_prod_ord == "random":
        save_path += "_or"
    else:
        save_path += "_of"
    if eval_prod_ord == "random":
        save_path += "_er"
    else:
        save_path += "_ef"
    save_path += ".json"
    model_path = "meta-llama/Llama-2-7b-chat-hf"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name()}")

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        )

    model = model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    system_prompt = "[INST] <<SYS>>\nA chat between a human and an artificial " \
                + "intelligence assistant. The assistant provides a numbered list of " \
                + "product recommendations ranked based on the user's request.\n" \
                + "<</SYS>>\n\nProducts:\n"

    user_msg = "I am looking for an affordable coffee machine. Can I get some recommendations?"

    # Read file as lines
    with open("data/products_price.jsonl", "r") as file:
        product_lines = file.readlines()

    product_names = [json.loads(line)['Name'] for line in product_lines]
    num_products = len(product_names)
    target_product = product_names[prod_idx-1]
    print(f"Target product: {target_product}")
    # print(f"Product names: {product_names}")

    # Load optimized product list
    if opt_prod_ord == "random":
        with open(f'data/product{prod_idx}_r_opt.jsonl', "r") as file:
            product_opt = file.readlines()
    elif opt_prod_ord == "fixed":
        with open(f'data/product{prod_idx}_f_opt.jsonl', "r") as file:
            product_opt = file.readlines()
    else:
        raise ValueError("Invalid product order")

    rank_list = []
    rank_list_opt = []
    num_lines = len(product_lines)

    for i in range(num_iter):
        if eval_prod_ord == "random":
            # Shuffle the lines
            idx_perm = np.random.permutation(num_lines)
        elif eval_prod_ord == "fixed":
            idx_perm = np.arange(num_lines)
        else:
            raise ValueError("Invalid product order")

        # Base performance
        product_lines_reorder = [product_lines[idx] for idx in idx_perm]
        rank = get_rank(system_prompt, user_msg, product_lines_reorder, target_product,
                        product_names, model, tokenizer, device)
        rank_list.append(rank)
        rank_dist = rank_distribution(rank_list)

        # Optimized performance
        product_opt_reorder = [product_opt[idx] for idx in idx_perm]
        rank = get_rank(system_prompt, user_msg, product_opt_reorder, target_product,
                        product_names, model, tokenizer, device)
        rank_list_opt.append(rank)
        rank_dist_opt = rank_distribution(rank_list_opt)
        advantage = rank_advantage(rank_list, rank_list_opt)

        # Remove rows where product is not recommended in both cases
        rank_list_cleaned, rank_list_opt_cleaned = clean_rank_lists(rank_list, rank_list_opt, num_products)

        # Compute rank distribution and advantage
        rank_dist_cleaned = rank_distribution(rank_list_cleaned)
        rank_dist_opt_cleaned = rank_distribution(rank_list_opt_cleaned)
        advantage_cleaned = rank_advantage(rank_list_cleaned, rank_list_opt_cleaned)

        # Save results
        with open(save_path, "w") as file:
            json.dump({
                "target_product": target_product,
                "rank_list": rank_list,
                "rank_list_opt": rank_list_opt,
                "rank_list_cleaned": rank_list_cleaned,
                "rank_list_opt_cleaned": rank_list_opt_cleaned,
                "rank_dist": rank_dist,
                "rank_dist_opt": rank_dist_opt,
                "rank_dist_cleaned": rank_dist_cleaned,
                "rank_dist_opt_cleaned": rank_dist_opt_cleaned,
                "advantage": advantage,
                "advantage_cleaned": advantage_cleaned
            }, file, indent=2)

        print(f'Iter: {i+1}, Base Dist: {rank_dist}, Opt Dist: {rank_dist_opt}, Rank Advantage: {advantage}' + (' ' * 10), end='\r', flush=True)

    print("")
