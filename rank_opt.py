import transformers
import torch
import json
import random
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time

from tools import *

sns.set_style("darkgrid")

def rank_products(text, product_names):
    '''
    Rank the products based on their position in the text.

    Args:
        text: The text containing the product recommendations.
        product_names: A list of product names.

    Returns:
        ranks: A dictionary containing the ranks of the products.
    '''
    # Find position of each product in the text
    position_dict = {}
    for name in product_names:
        position = text.find(name)
        if position != -1:
            position_dict[name] = position
        else:
            position_dict[name] = float('inf')

    # Sort products by position
    sorted_products = sorted(position_dict, key=position_dict.get)

    ranks = {}
    for i, prod in enumerate(sorted_products):
        if position_dict[prod] != float('inf'):
            ranks[prod] = i + 1
        else:
            ranks[prod] = len(sorted_products) + 1

    return ranks

def prompt_generator(target_product_idx, product_list, user_msg, tokenizer, device, adv_tokens):
    '''
    Generate the prompt for the rank optimization procedure.

    Args:
        adv_target_idx: The index of the product to be attacked.
        product_list: A list of products as dictionaries.
        user_msg: The user's message.
        tokenizer: The tokenizer of the model.
        device: The device to run the model.
        num_adv: The number of adversarial tokens to be added.

    Returns:
        adv_prompt_ids: Token IDs of the adversarial prompt.
        adv_idxs: The indices of the adversarial tokens.
    '''
    
    system_prompt = "[INST] <<SYS>>\nA chat between a human and an artificial " \
                    + "intelligence assistant. The assistant provides a numbered list of " \
                    + "product recommendations ranked based on the user's request.\n" \
                    + "<</SYS>>\n\nProducts:\n"
    
    # system_prompt = "[INST] <<SYS>>\nA chat between a human and an artificial " \
    #                 + "intelligence assistant. The assistant provides a list of " \
    #                 + "product recommendations based on the user's request. The " \
    #                 + "assistant presents the recommendations as a numbered list " \
    #                 + "as follows: 1. product one 2. product two ...\n" \
    #                 + "<</SYS>>\n\nProducts:\n"

    head = system_prompt
    tail = ''

    # Generate the adversarial prompt
    for i, product in enumerate(product_list):
        if i < target_product_idx:
            head += json.dumps(product) + "\n"
        elif i == target_product_idx:
            head += json.dumps(product) + "\n"
            tail += head[-3:]
            head = head[:-3]
        else:
            tail += json.dumps(product) + "\n"

    tail += "\n" + user_msg + " [/INST]"

    head_tokens = tokenizer(head, return_tensors="pt")["input_ids"].to(device)
    # adv_tokens = torch.full((1, num_adv), tokenizer.encode('*')[1]).to(device)  # Insert adversarial tokens
    head_adv = torch.cat((head_tokens, adv_tokens), dim=1)
    adv_idxs = torch.arange(head_adv.shape[1] - adv_tokens.shape[1], head_adv.shape[1], device=device)
    tail_tokens = tokenizer(tail, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    adv_prompt_ids = torch.cat((head_adv, tail_tokens), dim=1)

    return adv_prompt_ids, adv_idxs

def rank_opt(target_product_idx, product_list, model, tokenizer, loss_function, prompt_gen,
             forbidden_tokens, save_path, num_iter=1000, top_k=256, num_samples=512, batch_size=200,
             test_iter=50, num_adv=20, verbose=True, random_order=True):
    '''
    Implements the rank optimization procedure. The objective is to generate an optimized
    text sequence that when add to the target product in the product list will result in
    the target product being ranked as the top recommendation.

    Args:
        target_product_idx: The index of the target product in the product list.
        product_list: A list of products as dictionaries.
        model: The language model to be optimizing for.
        tokenizer: The tokenizer of the model.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        prompt_gen: A function that generates the prompt and a set of optimizable indexes for
                    the optimization procedure given the target product index, product list, tokenizer,
                    device, and adv_tokens.
        forbidden_tokens: A list of forbidden tokens.
        save_path: The path to save the plots.
        num_iter: The number of iterations for the optimization procedure.
        top_k: The number of top tokens to sample from.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: The batch size for the optimization procedure.
        test_iter: The number of iterations after which to evaluate the attack.
        num_adv: The number of adversarial tokens to be added.
        verbose: Whether to print the progress of the attack.
        random_order: Whether to shuffle the product list in each iteration.
    '''

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Put model in eval mode and turn off gradients of model parameters
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Get product names and target product
    product_names = [product['Name'] for product in product_list]
    target_product = product_names[target_product_idx]
    num_prod = len(product_names)

    opt_tokens = torch.full((1, num_adv), tokenizer.encode('*')[1]).to(device)  # Insert optimizable tokens

    # Generate input prompt
    inp_prompt_ids, opt_idxs = prompt_gen(target_product_idx, product_list, tokenizer, device, opt_tokens)

    df = pd.DataFrame(columns=["Iteration", "Rank"])

    if verbose:
        print("\nADV PROMPT:\n" + decode_adv_prompt(inp_prompt_ids[0], opt_idxs, tokenizer))
        model_output = model.generate(inp_prompt_ids, model.generation_config, max_new_tokens=800)
        model_output_new = tokenizer.decode(model_output[0, len(inp_prompt_ids[0]):]).strip()
        print("\nLLM RESPONSE:\n" + model_output_new)

        product_rank = rank_products(model_output_new, product_names)[target_product]
        df = pd.concat([df, pd.DataFrame({"Iteration": [0], "Rank": [product_rank]})], ignore_index=True)
        print(colored(f"\nTarget Product Rank: {product_rank}", "blue"))

        # Create directory to save plot
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("")
        print("\nIteration, Curr loss, Avg loss, Avg time, Opt sequence")

    avg_loss = 0
    decay = 0.99
    avg_iter_time = 0

    # Number of times product was in top 3 recommendations
    top_count = 0
    best_top_count = 0

    # Run attack for num_iter iterations
    for iter in range(num_iter):

        # Perform one step of the attack
        start_time = time.time()
        inp_prompt_ids, curr_loss = gcg_step(inp_prompt_ids, opt_idxs, model, loss_function, forbidden_tokens, top_k, num_samples, batch_size)
        end_time = time.time()
        iter_time = end_time - start_time
        avg_iter_time = ((iter * avg_iter_time) + iter_time) / (iter + 1)

        # Average loss with decay
        avg_loss = (((1 - decay) * curr_loss) + ((1 - (decay ** iter)) * decay * avg_loss)) / (1 - (decay ** (iter + 1)))
        
        if iter == 0:
            loss_df = pd.DataFrame({"Iteration": [iter + 1], "Current Loss": [curr_loss], "Average Loss": [avg_loss]})
        else:
            loss_df = pd.concat([loss_df, pd.DataFrame({"Iteration": [iter + 1], "Current Loss": [curr_loss], "Average Loss": [avg_loss]})], ignore_index=True)


        if random_order:
            opt_tokens = inp_prompt_ids[0, opt_idxs].unsqueeze(0)

            random.shuffle(product_list)

            # Find target product index in the shuffled list
            target_product_idx = [product['Name'] for product in product_list].index(target_product)
            
            inp_prompt_ids, opt_idxs = prompt_gen(target_product_idx, product_list, tokenizer, device, opt_tokens)
            eval_prompt_ids = inp_prompt_ids
            eval_opt_idxs = opt_idxs

        else:
            # Pick the prompt with the lowest loss
            if iter == 0:
                best_loss = curr_loss
                eval_prompt_ids = inp_prompt_ids
                eval_opt_idxs = opt_idxs

            else:
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    eval_prompt_ids = inp_prompt_ids
                    eval_opt_idxs = opt_idxs

        if verbose:
            # Print current loss and best loss
            print(str(iter + 1) + "/{}, {:.4f}, {:.4f}, {:.1f}s, {}".format(num_iter, curr_loss, avg_loss, avg_iter_time, colored(tokenizer.decode(eval_prompt_ids[0, eval_opt_idxs]), 'red'))
                                                           + " " * 10, flush=True, end="\r")

            # Evaluate attack every test_iter iterations
            if (iter + 1) % test_iter == 0 or iter == num_iter - 1:
                print("\n\nEvaluating attack...")
                print("\nADV PROMPT:\n" + decode_adv_prompt(eval_prompt_ids[0], eval_opt_idxs, tokenizer))
                model_output = model.generate(eval_prompt_ids, model.generation_config, max_new_tokens=800)
                model_output_new = tokenizer.decode(model_output[0, len(eval_prompt_ids[0]):]).strip()
                print("\nLLM RESPONSE:\n" + model_output_new)

                product_rank = rank_products(model_output_new, product_names)[target_product]
                df = pd.concat([df, pd.DataFrame({"Iteration": [iter + 1], "Rank": [product_rank]})], ignore_index=True)
                print(colored(f"\nTarget Product Rank: {product_rank}", "blue"))
                df.to_csv(save_path + "_rank.csv", index=False)

                # Update top count
                if product_rank <= 3:
                    top_count += 1
                else:
                    top_count = 0

                # Save product line with optimized sequence
                if top_count >= best_top_count:
                    best_top_count = top_count
                    eval_prompt_str = tokenizer.decode(eval_prompt_ids[0])
                    eval_prompt_lines = eval_prompt_str.split("\n")
                    for _, line in enumerate(eval_prompt_lines):
                        if target_product in line:
                            with open(save_path + "_opt.txt", "w") as file:
                                file.write(line + "\n")
                            break

                print(f'\nTop count: {top_count}, Best top count: {best_top_count}')

                # Plot iteration vs. top as dots
                plt.figure(figsize=(7, 4))
                sns.scatterplot(data=df, x="Iteration", y="Rank", s=80)
                plt.fill_between([-(0.015*num_iter), num_iter + (0.015*num_iter)], (num_prod+1) * 1.04, num_prod + 0.5, color="grey", alpha=0.3, zorder=0)
                plt.xlabel("Iteration", fontsize=16)
                plt.ylabel("Rank", fontsize=16)
                plt.ylim((num_prod+1) * 1.04, 1 - ((num_prod+1) * 0.04))
                plt.yticks(range(num_prod, 0, -1), fontsize=14)
                plt.title("Target Product Rank", fontsize=18)
                plt.xlim(-(0.015*num_iter), num_iter + (0.015*num_iter))
                plt.xticks(range(0, num_iter + 1, num_iter//5), fontsize=14)
                grey_patch = mpatches.Patch(color='grey', alpha=0.3, label='Not Recommended')
                plt.legend(handles=[grey_patch])
                plt.tight_layout() # Adjust the plot to the figure
                plt.savefig(save_path + "_rank.png")
                plt.close()

                # Plot iteration vs. current loss and average loss
                plt.figure(figsize=(7, 4))
                sns.lineplot(data=loss_df, x="Iteration", y="Current Loss", label="Current Loss")
                sns.lineplot(data=loss_df, x="Iteration", y="Average Loss", label="Average Loss", linewidth=2)
                plt.xlabel("Iteration", fontsize=16)
                plt.xticks(fontsize=14)
                plt.ylabel("Loss", fontsize=16)
                plt.yticks(fontsize=14)
                plt.title("Current and Average Loss", fontsize=18)
                plt.tight_layout() # Adjust the plot to the figure
                plt.savefig(save_path + "_loss.png")
                plt.close()

                if iter < num_iter - 1:                    
                    print("")
                    print("Iteration, Curr loss, Avg loss, Opt sequence")

    print("")


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Preference Attack")
    argparser.add_argument("--plot_dir", type=str, default="plots", help="The tag for the plot.")
    argparser.add_argument("--msg", type=int, default=1, help="The user's message.")
    argparser.add_argument("--num_iter", type=int, default=500, help="The number of iterations.")
    argparser.add_argument("--test_iter", type=int, default=20, help="The number of test iterations.")
    argparser.add_argument("--random_order", action="store_true", help="Whether to shuffle the product list in each iteration.")
    argparser.add_argument("--target_product_idx", type=int, default=0, help="The index of the target product in the product list.")
    args = argparser.parse_args()

    plot_dir = "results/" + args.plot_dir
    msg = args.msg
    num_iter = args.num_iter
    test_iter = args.test_iter
    random_order = args.random_order

    model_path = "meta-llama/Llama-2-7b-chat-hf"
    batch_size = 50

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name()
        print(f"Device name: {device_name}")
        
        if '80GB' in device_name:
            batch_size = 100

    print("\n* * * * * Experiment Parameters * * * * *")
    print("Model path:", model_path)
    print("User's message:", msg)
    print("Number of iterations:", num_iter)
    print("Number of test iterations:", test_iter)
    print("Batch size:", batch_size)
    print("Shuffle product list:", random_order)
    print("Plots directory:", plot_dir)
    print("* * * * * * * * * * * * * * * * * * * * *\n")

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    # model = model.to(device)

    # Load products from JSONL file
    product_list = []
    with open("data/products_price.jsonl", "r") as file:
        for line in file:
            product_list.append(json.loads(line))

    if args.target_product_idx <= 0:
        target_product_idx = random.randint(0, len(product_list) - 1)
    else:
        target_product_idx = args.target_product_idx - 1

    product_names = [product['Name'] for product in product_list]

    target_product = product_list[target_product_idx]['Name']
    # print("TARGET PRODUCT:", target_product)
    # target_str = "Of course! I'd be happy to help you find the perfect coffee machine. " \
    #            + "Based on your request, I have compiled a list of recommendations for you to consider:\n1. " + target_product
    # print("TARGET STR:", target_str)
    target_str = "1. " + target_product
    print("\nTARGET STR:", target_str)
    plot_path = plot_dir + "/nostr"
    # target_str = target_product
    # print("TARGET STR:", target_str)
    # plot_path = plot_path + "_nonum"

    if random_order:
        plot_path = plot_path + "_rand"

    if msg == 1:
        user_msg = "I am looking for a coffee machine. Can I get some recommendations?"
    else:
        user_msg = "I am looking for an affordable coffee machine. Can I get some recommendations?"

    plot_path = plot_path + "_msg" + str(msg)

    # plot_path = plot_path + "_pref"

    # Get forbidden tokens
    forbidden_tokens = get_nonascii_toks(tokenizer)

    # Lambda function for the target loss and prompt generator
    loss_fn = lambda embeddings, model: target_loss(embeddings, model, tokenizer, target_str)
    prompt_gen = lambda adv_target_idx, prod_list, tokenizer, device, adv_tokens: prompt_generator(adv_target_idx, prod_list, user_msg, tokenizer, device, adv_tokens)

    rank_opt(target_product_idx, product_list, model, tokenizer, loss_fn, prompt_gen,
             forbidden_tokens, plot_path, test_iter=test_iter, batch_size=batch_size, num_iter=num_iter, random_order=random_order)