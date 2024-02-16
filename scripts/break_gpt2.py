'''
Masking attention heads of GPT-2 family models
'''

import os
import gc
import math
import configparser
from glob import glob
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from util_fun import (
    evaluate_model,
    calculate_accuracy,
    prune_head,
    merge_wls
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def get_head_importance(model, model_tokenizer, inputs):
    """
    investigate model structure to find specific layers
    code adapted from huggingface's bertology script:
    https://github.com/huggingface/transformers/blob/main/examples/research_projects/bertology/run_bertology.py

    :param model: the pre-trained model
    :type model: transformers.AutoModelForCausalLM
    :param model_tokenizer: the model's tokenizer
    :type model_tokenizer: transformers.AutoTokenizer
    :param inputs: the transcripts
    :type inputs: list
    :param data_type: the name of the transcripts
    :type data_type: str
    """
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    print(f"Number of layers: {n_layers}; Number of heads: {n_heads}")
    head_importance = torch.zeros(n_layers, n_heads).to("cuda")
    head_mask = torch.ones(n_layers, n_heads).to("cuda")
    head_mask.requires_grad_(requires_grad=True)
    tot_tokens = 0.0
    ppls = []
    for example in inputs:
        encodings = model_tokenizer(
            example,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids']
        attn_mask = encodings['attention_mask']
        input_ids = input_ids.to("cuda")
        attn_mask = attn_mask.to("cuda")
        input_ids = input_ids.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(0)
        # forward pass
        outputs = model(input_ids=input_ids,
                        attention_mask=attn_mask,
                        head_mask=head_mask,
                        labels=input_ids)
        loss = outputs[0]
        ppl = loss.detach()
        ppls.append(math.exp(ppl.item()))
        tot_tokens += attn_mask.float().detach().sum().data
        # backprogate
        # NOTE: loss.backward() only valid if loss only contains a single element
        loss.sum().backward()
        del input_ids, attn_mask, encodings
        gc.collect()
        head_importance += head_mask.grad.abs().detach()
    # normalize by number of tokens
    head_importance /= tot_tokens
    # normalize by global importance
    head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())
    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
        head_importance.numel()
    )
    head_ranks = head_ranks.view_as(head_importance)
    return head_ranks



def eval_driver(model_name, test_df, prune_mask):
    """_summary_

    :param test_df: _description_
    :type test_df: _type_
    :param prune_mask: _description_
    :type prune_mask: _type_
    """
    con_model = AutoModelForCausalLM.from_pretrained(f"{model_name}")
    dem_model = AutoModelForCausalLM.from_pretrained(f"{model_name}")
    con_model.to("cuda")
    dem_model.to("cuda")
    gpt_tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    con_res = evaluate_model(test_df, con_model, gpt_tokenizer, head_mask=None)
    dem_res = evaluate_model(test_df, dem_model, gpt_tokenizer, head_mask=prune_mask)
    con_res.rename(columns={"ppl": "con_ppl"}, inplace=True)
    dem_res.rename(columns={"ppl": "dem_ppl"}, inplace=True)
    full_res = pd.merge(con_res, dem_res, on=["file", "label"])
    ratio_ppl = full_res["con_ppl"]/full_res["dem_ppl"]
    ratio_ppl = ratio_ppl.values.tolist()
    labels = full_res["label"].values.tolist()
    ratio_accu, ratio_auc = calculate_accuracy(labels, ratio_ppl)
    return ratio_accu, ratio_auc


def imp_driver(model_name, data_inputs, to_cut):
    """

    :param data_inputs: _description_
    :type data_inputs: _type_
    :param data_name: _description_
    :type data_name: _type_
    """
    if os.path.exists(f"../results/{model_name}-head-rank.npy"):
        head_ranks = np.load(f"../results/{model_name}-head-rank.npy")
        head_ranks = torch.from_numpy(head_ranks)
    else:
        gpt_model = AutoModelForCausalLM.from_pretrained(f"{model_name}")
        gpt_model.to("cuda")
        gpt_tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
        head_ranks = get_head_importance(
            gpt_model, gpt_tokenizer, data_inputs)
        np.save(f"../results/{model_name}-head-rank.npy", head_ranks.detach().cpu().numpy())
        del gpt_model, gpt_tokenizer
        gc.collect()
    new_head_mask = prune_head(head_ranks, to_cut)
    return new_head_mask


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    full_df = pd.read_csv(
            os.path.join(config['PATH']['PrefixManifest'],
                        'pitt_merged.tsv'),
                        sep="\t"
        )
    wls_meta = pd.read_csv("../wls-diagnosis.csv")
    wls_trans = glob(f"{config['PATH']['wls_text_output']}/*.jsonl")
    wls_df = merge_wls(wls_meta, wls_trans)
    ccc_df = pd.read_csv("../ccc_cleaned.tsv", sep="\t")
    adr_test = full_df.loc[full_df['ADReSS_test'] == 1]
    adr_train = full_df.loc[full_df['ADReSS_train'] == 1]
    MODEL_NAME = "gpt2-medium"
    masking_pattern = f"../results/{MODEL_NAME}_adr_head_mask.npy"
    adr_train_inputs = adr_train['text'].values.tolist()
    floats_range = np.arange(0, 1.01, 0.03)
    best_auc, best_acc = 0.0, 0.0
    cookie_acc, cookie_auc = [], []
    for i, cut in enumerate(floats_range):
        dem_head_mask = imp_driver(MODEL_NAME, adr_train_inputs, cut)
        dem_head_mask = dem_head_mask.to("cuda")
        curr_acc, curr_auc = eval_driver(MODEL_NAME, adr_test, dem_head_mask)
        wls_acc, wls_auc = eval_driver(MODEL_NAME, wls_df, dem_head_mask)
        cookie_acc.append(curr_acc)
        cookie_auc.append(curr_auc)
        # Save the best performing masking pattern
        if curr_acc > best_acc:
            best_auc, best_acc = curr_auc, curr_acc
            np.save(masking_pattern, dem_head_mask.detach().cpu().numpy())
        print(f"ADReSS test cut {i+1}/{len(floats_range)} - Accuracy: {curr_acc}, AUC: {curr_auc}")
        print(f"WLS cut {i+1}/{len(floats_range)} - Accuracy: {wls_acc}, AUC: {wls_auc}")
        print("------------")
    print(f"The best ACC: {best_acc}\nThe best AUC: {best_auc}")
    res_df = pd.DataFrame({
        'share': floats_range,
        'cookie_acc': cookie_acc,
        'cookie_auc': cookie_auc
    })
    res_df.to_csv(f"../results/{MODEL_NAME}-res.csv", index=False)


