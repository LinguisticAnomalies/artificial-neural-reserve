'''
utility functions
'''
import gc
import math
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc


def print_2d_tensor(tensor):
    """Print a 2D tensor"""
    print(f"lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def merge_wls(meta_df, trans_list):
    hc_trans = get_hc_pars(meta_df, trans_list)
    hc_df = pd.DataFrame(hc_trans)
    hc_df["label"] = [0]*len(hc_df)
    dem_trans = get_dem_pars(meta_df, trans_list)
    dem_df = pd.DataFrame(dem_trans)
    dem_df["label"] = [1]*len(dem_df)
    wls_df = pd.concat([hc_df, dem_df])
    wls_df = wls_df.sample(frac=1)
    return wls_df


def get_dem_pars(meta_df, trans_list):
    """
    get the transcripts from WLS dementia participants

    :param meta_df: _description_
    :type meta_df: _type_
    :param trans_list: _description_
    :type trans_list: _type_
    :return: _description_
    :rtype: _type_
    """
    # dementia diagnosis via consenus
    meta_df = meta_df.loc[meta_df['Level of cognitive impairment via Consensus'] == 3]
    dem_pids = meta_df['idtlkbnk'].astype(str).tolist()
    wls_trans = []
    for tran_file in trans_list:
        pid = os.path.splitext(os.path.basename(tran_file))[0]
        # add prefix
        pid = f"20000{pid}"
        if pid in dem_pids:
            with open(tran_file, "r") as read_file:
                trans = [json.loads(line).get("text") for line in read_file]
                trans = ". ".join(trans) + "."
                wls_trans.append({"file": pid, "text": trans})
    return wls_trans


def get_hc_pars(meta_df, trans_list):
    """
    get the transcripts from WLS healthy participants

    :param meta_df: the WLS metadata
    :type meta_df: pd.DataFrame
    :param trans_list: the path to utterances of WLS
    :type trans_list: list
    :return: the transcripts from WLS healthy participants
    :rtype: list
    """
    # no dementia diagnosis via consenus
    meta_df = meta_df.loc[meta_df['Level of cognitive impairment via Consensus'] == 1]
    # no MCI diagnosis
    meta_df = meta_df.loc[meta_df['MCI subtype'] == -2]
    # no previous stroke history
    meta_df = meta_df.loc[meta_df['Has a doctor ever told R they had a stroke?'] == 2]
    hc_pids = meta_df['idtlkbnk'].astype(str).tolist()
    wls_trans = []
    for tran_file in trans_list:
        pid = os.path.splitext(os.path.basename(tran_file))[0]
        # add prefix
        pid = f"20000{pid}"
        if pid in hc_pids:
            with open(tran_file, "r") as read_file:
                trans = [json.loads(line).get("text") for line in read_file]
                trans = ". ".join(trans) + "."
                wls_trans.append({"file": pid, "text": trans})
    return wls_trans


def prune_head(head_ranks, threshold):
    """
    prune attention heads by ranking

    :param head_importance: the head importance estimated from model fine-tuning
    :type model: torch.Tensor
    :param head_ranks: the head importance ranking
    :type head_ranks: torch.Tensor
    :param threshold: the % of attention heads to be masked
    :type threshold: float
    """
    new_head_mask = torch.ones_like(head_ranks)
    num_to_mask = max(1, int(new_head_mask.numel() * threshold))
    print(f"Number of attention heads to be pruned: {num_to_mask}, {threshold*100}%")
    top_threshold_index = max(1, int(num_to_mask * 0.5))
    sorted_head_ranks, _ = torch.sort(head_ranks.view(-1))
    top_threshold = sorted_head_ranks[top_threshold_index]
    mask_most_important = head_ranks <= top_threshold
    new_head_mask[mask_most_important] = 0.0
    bottom_threshold_index = -max(1, int(num_to_mask * 0.5))
    bottom_threshold = sorted_head_ranks[bottom_threshold_index]
    mask_least_important = head_ranks > bottom_threshold
    new_head_mask[mask_least_important] = 0.0
    return new_head_mask


def evaluate_model(test_frame, model, tokenizer, head_mask=None):
    """
    estimate perplexities on the test dataset

    :param test_frame: the test dataset
    :type test_frame: pd.DataFrame
    :param model: the pre-trained model
    :type model: transformers.AutoModelForCausalLM
    :param tokenizer: the model's tokenizer
    :type tokenizer: transformers.AutoTokenizer
    :param head_mask: the head mask for the dementia model, defaults to None
    :type head_mask: torch.Tensor, optional
    """
    model.eval()
    res_list = []
    for _, row in test_frame.iterrows():
        trans = row["text"]
        encodings = tokenizer(
            trans, return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
            truncation=True,
            max_length=1024)
        input_ids = encodings["input_ids"]
        att_mask = encodings["attention_mask"]
        input_ids = input_ids.to("cuda")
        att_mask = att_mask.to("cuda")
        outputs = model(
            input_ids, attention_mask=att_mask,
            labels=input_ids,
            head_mask=head_mask)
        perp = math.exp(outputs[0].item())
        eval_dict = {"file": row["file"],
                     "label": row["label"],
                     "ppl": perp}
        res_list.append(eval_dict)
        del input_ids, att_mask, outputs
        gc.collect()
    res_df = pd.DataFrame(res_list)
    return res_df


def calculate_accuracy(labels, perp):
    """
    calculate accuracy given labels and perpelxity scores at equal error rate

    :param labels: the transcript labels
    :type labels: list
    :param perp: the perplexity scores
    :type perp: list
    :return: accuracy and auc
    """
    fpr, tpr, _ = roc_curve(labels, perp)
    fnr = 1 - tpr
    tnr = 1 - fpr
    auc_level = auc(fpr, tpr)
    prevalence = np.count_nonzero(labels)/len(labels)
    eer_point = np.nanargmin(np.absolute((fnr - fpr)))
    tpr_at_eer = tpr[eer_point]
    tnr_at_eer = tnr[eer_point]
    accuracy = tpr_at_eer * prevalence + tnr_at_eer * (1-prevalence)
    return round(accuracy, 2), round(auc_level, 2)
