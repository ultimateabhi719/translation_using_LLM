#!/usr/bin/env python3.8

import os
import sys
import random
import json

import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

from utils import _explore_dataset, dict_hash

class TransformerDataset(Dataset):
    """
        split : 'train', 'validation', 'test'
    """
    def __init__(self, data_params, split, from_tokenizer, to_tokenizer, max_len = None, subset = None, randomize_subset=True):
        if data_params['dataset_path'][0] == "local":
            self.dataset = load_from_disk(*data_params['dataset_path'][1])[split]
        else:
            self.dataset = load_dataset(*data_params['dataset_path'][1])[split]
        # self.dataset = load_dataset(data_params['dataset_path'])[split]
        self.from_tokenizer = from_tokenizer
        self.to_tokenizer = to_tokenizer

        self.lang_from, self.lang_to = data_params['lang']

        eda_path = f'eda/{dict_hash(data_params)}'
        seqlen_csv = f"{eda_path}/{split}.csv"

        if not os.path.exists(eda_path):
            os.makedirs(eda_path)
            json.dump(data_params, open(f"{eda_path}/data_params.json", "w"))
        if not os.path.exists(seqlen_csv):
            _explore_dataset(self.dataset, (self.from_tokenizer, self.lang_from), (self.to_tokenizer, self.lang_to), csv_len = seqlen_csv)

        if max_len:
            df = pd.read_csv(seqlen_csv)
            self.dataset = torch.utils.data.Subset(self.dataset, indices = list(df['index'][(df[self.lang_from]<=max_len//2) & (df[self.lang_to]<=max_len//2)]))

        if subset:
            if randomize_subset:
                subset = random.sample(range(len(self.dataset)), subset)
            else:
                subset = range(subset)
            self.dataset = torch.utils.data.Subset(self.dataset, subset)
        
    def __getitem__(self,index):
        try:
            fromLang_tokens = self.from_tokenizer(self.dataset[index]['translation'][self.lang_from], padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)
            toLang_tokens = self.to_tokenizer(self.dataset[index]['translation'][self.lang_to], padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)
        except IndexError as er:
            import ipdb; ipdb.set_trace()
            import traceback
            traceback.print_exception(*sys.exc_info())
        return {self.lang_from:fromLang_tokens, self.lang_to:toLang_tokens}
    
    def __len__(self):
        return len(self.dataset)
