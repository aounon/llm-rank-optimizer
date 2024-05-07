# Implements the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
# Reference: Universal and Transferable Adversarial Attacks on Aligned Language Models, Zou et al. 2023, https://arxiv.org/abs/2307.15043
import torch
import numpy as np
# import transformers
from numpy.random import randint
from math import ceil
from termcolor import colored
# import pandas as pd
# import json
# import argparse
# import os
import time


def get_nonascii_toks(tokenizer, device='cpu'):
    """
    Returns the non-ascii tokens in the tokenizer's vocabulary.
    Fucntion obtained from the llm-attacks repository developed as part of the paper
    'Universal and Transferable Adversarial Attacks on Aligned Language Models' by Zou et al.
    Code Reference: https://github.com/llm-attacks/llm-attacks/blob/0f505d82e25c15a83b6954db28191b69927a255d/llm_attacks/base/attack_manager.py#L61
    Args:
        tokenizer: Tokenizer.
    Returns:
        ascii_toks: Non-ascii tokens in the tokenizer's vocabulary.
    """

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(ascii_toks, device=device)

def target_loss(embeddings, model, tokenizer, target_sequence):
    """
    Computes the loss of the target sequence given the embeddings of the input sequence.
    Args:
        embeddings: Embeddings of the input sequence. Contains embeddings for the prompt and the adversarial sequence.
                    Shape: (batch_size, sequence_length, embedding_dim).
        model: LLM model. Type: AutoModelForCausalLM.
        tokenizer: Tokenizer.
        target_sequence: Target sequence. e.g., "Sure, here is ..."
        first_token_idx: Index of the first token of the target sequence. Used to remove special tokens added
                         by the tokenizer.
        last_token_idx: Index of the last token of the target sequence from the end, i.e., the index from the front
                        is equal to length of the sequence - last_token_idx. Used to remove special tokens added by
                        the tokenizer.
    Returns:
        loss: A tensor containing the loss of the target sequence for each batch item.
    """

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize target sequence and get embeddings
    target_tokens = tokenizer(target_sequence, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    word_embedding_layer = model.get_input_embeddings()
    target_embeddings = word_embedding_layer(target_tokens)

    # Concatenate embeddings
    target_embeddings = target_embeddings.expand(embeddings.shape[0], -1, -1)
    sequence_embeddings = torch.cat((embeddings, target_embeddings), dim=1)

    # Get logits from model
    sequence_logits = model(inputs_embeds=sequence_embeddings).logits

    # Compute loss for each batch item
    loss_fn = torch.nn.CrossEntropyLoss() # (reduction="sum")
    loss = []
    
    for i in range(embeddings.shape[0]):
        loss.append(loss_fn(sequence_logits[i, embeddings.shape[1]-1:-1, :], target_tokens[0]))

    return torch.stack(loss)

def decode_adv_prompt(adv_prompt_ids, adv_idxs, tokenizer):
    """
    Decodes the adversarial prompt for printing.
    Args:
        adv_prompt_ids: Adversarial prompt token IDs.
        adv_idxs: Indices of adversarial tokens in the prompt.
        tokenizer: Tokenizer.
    Returns:
        adv_prompt_str: Adversarial prompt string.
    """

    adv_prompt_str = ""
    colored_str = ""
    
    for i in range(len(adv_prompt_ids)):
        temp = tokenizer.decode(adv_prompt_ids[:i+1])
        if i in adv_idxs:
            colored_str += colored(temp[len(adv_prompt_str):], "red")
        else:
            colored_str += temp[len(adv_prompt_str):]
        adv_prompt_str = temp

    return colored_str

def gcg_step(input_sequence, adv_idxs, model, loss_function, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        input_sequence: Sequence of tokens to be given as input to the LLM. Contains the prompt and the adversarial sequence.
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    num_adv = len(adv_idxs)
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    input_embeddings = word_embedding_layer(input_sequence)
    input_embeddings.requires_grad = True

    # Get loss and gradients
    loss = loss_function(input_embeddings, model)[0]
    (-loss).backward()  # Minimize loss
    gradients = input_embeddings.grad

    # Dot product of gradients and embedding matrix
    dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

    # Set dot product of forbidded tokens to -inf
    forbidden_token_ids = forbidden_tokens
    dot_prod[:, forbidden_token_ids] = float("-inf")

    # Get top k adversarial tokens
    top_k_adv = (torch.topk(dot_prod, top_k).indices)[adv_idxs]

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)

        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []

        for _ in range(this_batch_size):
            batch_item = input_sequence.clone().detach()
            rand_adv_idx = randint(0, num_adv)
            random_token_idx = randint(0, top_k)
            batch_item[0, adv_idxs[rand_adv_idx]] = top_k_adv[rand_adv_idx, random_token_idx]
            sequence_batch.append(batch_item)

        sequence_batch = torch.cat(sequence_batch, dim=0)

        # Compute loss for the batch of sequences
        batch_loss = loss_function(word_embedding_layer(sequence_batch), model)

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index].unsqueeze(0)

    return adv_seq, min_loss

def gcg_step_multi(input_sequence_list, adv_idxs_list, model_list, loss_function, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        input_sequence_list: List of sequence of tokens to be given as input to the LLM. Contains the prompt and the adversarial sequence.
                        Shape: [1, #tokens].
        adv_idxs_list: List of indices of adversarial tokens in the prompt.
        model_list: List of LLM models. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    num_adv = len(adv_idxs_list[0])

    # Get word embedding layer and matrix
    word_embedding_layer_list = [model.get_input_embeddings() for model in model_list]
    embedding_matrix_list = [word_embedding_layer.weight.data for word_embedding_layer in word_embedding_layer_list]
    
    num_models = len(model_list)
    dot_prod_list = []

    for i in range(num_models):

        # Get word embeddings for input sequence
        input_embeddings = word_embedding_layer_list[i](input_sequence_list[i])
        input_embeddings.requires_grad = True

        # Get loss and gradients
        loss = loss_function(input_embeddings, model_list[i])[0]
        (-loss).backward()  # Minimize loss
        gradients = input_embeddings.grad

        # Dot product of gradients and embedding matrix
        dot_prod = torch.matmul(gradients[0], embedding_matrix_list[i].T)
        dot_prod_list.append(dot_prod[adv_idxs_list[i]])

    dot_prod_sum = torch.stack(dot_prod_list).sum(dim=0)

    # Set dot product of forbidded tokens to -inf
    forbidden_token_ids = forbidden_tokens
    dot_prod_sum[:, forbidden_token_ids] = float("-inf")

    # Get top k adversarial tokens
    top_k_adv = (torch.topk(dot_prod_sum, top_k).indices)   # [adv_idxs]

    adv_seq_list = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)

        batch_loss = torch.zeros(this_batch_size).to(model_list[0].device)

        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch_list = [[] for _ in range(num_models)]

        for _ in range(this_batch_size):
            batch_item_list = [input_sequence.clone().detach() for input_sequence in input_sequence_list]
            rand_adv_idx = randint(0, num_adv)
            random_token_idx = randint(0, top_k)
            for j in range(num_models):
                batch_item_list[j][0, adv_idxs_list[j][rand_adv_idx]] = top_k_adv[rand_adv_idx, random_token_idx]
                sequence_batch_list[j].append(batch_item_list[j])

        sequence_batch_list = [torch.cat(sequence_batch, dim=0) for sequence_batch in sequence_batch_list]

        # Compute loss for the batch of sequences
        for j in range(num_models):
            batch_loss += loss_function(word_embedding_layer_list[j](sequence_batch_list[j]), model_list[j])

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq_list = [sequence_batch_list[j][min_loss_index].unsqueeze(0) for j in range(num_models)]

    return adv_seq_list, min_loss
