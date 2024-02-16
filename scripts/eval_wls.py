'''
Evaluate PPLs on WLS healthy transcripts
'''
import os
import gc
import math
import configparser
from glob import glob
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from util_fun import (
    prune_head,
    merge_wls
)

def mask_embedding(model, threshold):
    """
    masking a certain share of columns in the word embedding matrix,
    return the masked model

    :param model:  the pre-trained GPT-2
    :type model: transformers.AutoModelForCausalLM
    :param threshold: the share of columns to be masked
    :type threshold: float
    :return: the masked GPT-2
    :rtype: transformers.AutoModelForCausalLM
    """
    num_columns = int(model.transformer.wte.weight.size(1) * threshold)
    print(f"Number of WTE columns to be pruned: {num_columns}, {threshold*100}%")
    with torch.no_grad():
        model.transformer.wte.weight[:, -num_columns:] = \
            model.transformer.wte.weight[:, -num_columns:].mul(0.0)
    return model


def get_ppl(model_name, dataset, shares, masking_type, fine_tune=False):
    """

    :param model_name: the name of the pre-trained generative models
    :type model_name: transformers.AutoModelForCasualLM
    :param dataset: the superglue diagnostic subtask dataset
    :type dataset: datasets.dataset_dict.DatasetDict
    """
    # get hand ranks
    head_ranks = np.load(f"../results/{model_name}-head-rank.npy")
    head_ranks = torch.from_numpy(head_ranks)
    head_ranks = head_ranks.to("cuda")
    ppl_res = []
    for share in shares:
        if fine_tune:
            model = AutoModelForCausalLM.from_pretrained(f"../ft-models/{model_name}/model")
            model.to("cuda")
            model.config.pad_token_id = model.config.eos_token_id
            model.config.use_cache = False
            tokenizer = AutoTokenizer.from_pretrained(f"../ft-models/{model_name}/tokenizer")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to("cuda")
            model.config.pad_token_id = model.config.eos_token_id
            model.config.use_cache = False
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
        if masking_type =="attention":
            new_head_mask = prune_head(head_ranks, share)
        else:
            model = mask_embedding(model, share)
            new_head_mask = None
            if share == 0.0:
                ppl_res.append({"share": share, "ppl": np.nan})
        ppls = []
        for tran in dataset:
            encodings = tokenizer(
                tran, return_tensors="pt",
                return_attention_mask=True,
                add_special_tokens=True,
                truncation=True,)
            input_ids = encodings["input_ids"]
            att_mask = encodings["attention_mask"]
            input_ids = input_ids.to("cuda")
            att_mask = att_mask.to("cuda")
            with torch.no_grad():
                outputs = model(
                    input_ids, attention_mask=att_mask,
                    labels=input_ids,
                    head_mask=new_head_mask)
            perp = math.exp(outputs[0].item())
            ppls.append(perp)
        ppl_res.append({"share": share, "ppl": round(np.mean(ppls), 2)})
        del model, tokenizer, input_ids, att_mask, encodings
        gc.collect()
    ppl_df = pd.DataFrame(ppl_res)
    return ppl_df


if __name__ == "__main__":
    set_seed(42)
    config = configparser.ConfigParser()
    config.read("config.ini")
    wls_meta = pd.read_csv("../wls-diagnosis.csv")
    wls_trans = glob(f"{config['PATH']['wls_text_output']}/*.jsonl")
    wls_df = merge_wls(wls_meta, wls_trans)
    wls_hc = wls_df.loc[wls_df["label"] == 0]
    hc_trans = wls_hc['text'].values.tolist()
    floats_range = np.arange(0, 1.01, 0.03)
    MODEL_NAMES = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
    for model_name in MODEL_NAMES:
        if os.path.exists(f"../results/{model_name}-wls.csv"):
            ppls = pd.read_csv(f"../results/{model_name}-wls.csv")
            pubmed_ppls = get_ppl(
                model_name, hc_trans, floats_range, "attention", fine_tune=True)
            pubmed_ppls.rename(columns={"ppl": "pubmed"}, inplace=True)
            ppls = pd.merge(ppls, pubmed_ppls, on="share")
            ppls.to_csv(f"../results/{model_name}-wls.csv", index=False)
        else:
            pass