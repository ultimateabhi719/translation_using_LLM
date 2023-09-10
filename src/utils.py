#!/usr/bin/env python
# coding: utf-8

import os
import hashlib
import json
from typing import Dict, Any
import itertools
import pandas as pd

import torch


def _explore_dataset(dataset, from_tokenizer, to_tokenizer, csv_len = None):
    if csv_len is not None and os.path.exists(csv_len):
        return pd.read_csv(csv_len)

    from_tokenizer, lang_from = from_tokenizer
    to_tokenizer, lang_to = to_tokenizer

    df = pd.DataFrame.from_dict(dataset)

    def num_tokens(x):
        fromLang_len = from_tokenizer(x[lang_from], padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)['input_ids'].shape[1]
        toLang_len = to_tokenizer(x[lang_to], padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)['input_ids'].shape[1]
        return fromLang_len, toLang_len

    df[lang_from], df[lang_to] = zip(*df.translation.progress_apply(num_tokens))
    df['sum_len'] = df[lang_from]+df[lang_to]
    df = df[[lang_from, lang_to,'sum_len']].sort_values('sum_len', ascending=False)
    df['index'] = df.index
    if csv_len:
        df.to_csv(csv_len, index=False)
    return df


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def collate_tokens(batch, get_from_tokens, get_to_tokens):

    from_tokenizer, lang_from = get_from_tokens
    to_tokenizer, lang_to = get_to_tokens

    out = {}
    out['lang_from'] = {}
    out['lang_to'] = {}

    def _collate(batch, key1, key2, fillvalue = 0):
        return torch.tensor(list(zip(*itertools.zip_longest(*[b[key1][key2][0] for b in batch], fillvalue=fillvalue))))
    
    out['lang_from']['input_ids'] = _collate(batch, lang_from, 'input_ids', fillvalue = from_tokenizer.pad_token_id)
    out['lang_from']['attention_mask'] = _collate(batch, lang_from, 'attention_mask')

    out['lang_to']['input_ids'] =  _collate(batch, lang_to, 'input_ids', fillvalue = to_tokenizer.pad_token_id)
    out['lang_to']['attention_mask'] =  _collate(batch, lang_to, 'attention_mask')

    return out


def save_model(x0, model, optimizer, scheduler, save_path):
    torch.save({
            'x0': x0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict()
            }, save_path)


