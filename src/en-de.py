#!/usr/bin/env python
# coding: utf-8

import pandas as pd

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from tqdm.auto import tqdm

from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, BertTokenizer, EncoderDecoderModel

## Data
dataset = load_dataset(*['wmt14', 'de-en'])
subset = list(range(0, 8))
dataset['train'] = torch.utils.data.Subset(dataset['train'], subset)
dataset['validation'] = torch.utils.data.Subset(dataset['validation'], subset)
dataset['test'] = torch.utils.data.Subset(dataset['test'], subset)

BS = 2
train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=BS, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset['validation'], batch_size=BS, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=BS, shuffle=False)


## Setup Model

### Tokenizers
de_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
en_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

### Model & Loss

def compute_loss(predictions, targets):
    """Compute our custom loss"""
    criterion = nn.NLLLoss(ignore_index=de_tokenizer.pad_token_id)

    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]

    rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
    rearranged_target = targets.contiguous().view(-1)

    loss = criterion(rearranged_output, rearranged_target)

    return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "dbmdz/bert-base-german-cased")

model.config.decoder_start_token_id = de_tokenizer.cls_token_id
model.config.pad_token_id = de_tokenizer.pad_token_id
# model.config.eos_token_id = de_tokenizer.eos_token_id
# model.config.bos_token_id = de_tokenizer.bos_token_id
model = model.to(device)


## Train

optimizer = transformers.AdamW(model.parameters(), lr=1e-5)

def train_model(train_loader, epoch):
    model.train()
    epoch_loss = 0

    num_train_batches = len(train_loader)
    for i, b in tqdm(enumerate(train_loader), desc=f'epoch {epoch}', leave=False):

        optimizer.zero_grad()

        de_token = de_tokenizer(b['translation']['de'], padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)
        en_token = en_tokenizer(b['translation']['en'], padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)
        
        de_output = de_token['input_ids'].to(device)
        de_masks = de_token['attention_mask'].to(device)
        
        en_input = en_token['input_ids'].to(device)
        en_masks = en_token['attention_mask'].to(device)

        # decoder_input_ids = shift_tokens_right(de_token['input_ids'], model.config.pad_token_id, model.config.decoder_start_token_id).to(device)
        out = model(input_ids=en_input, attention_mask = en_masks, decoder_input_ids=de_output,labels=de_output,decoder_attention_mask=de_masks)
        prediction_scores = out[1]
        predictions = F.log_softmax(prediction_scores, dim=-1)
        loss = compute_loss(predictions, de_output)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / num_train_batches


for epoch in range(200):
    train_epoch_loss = train_model(train_loader, epoch)
    print("epoch {epoch:>{width}}: train_loss {train_loss:.3f}".format(epoch=epoch,train_loss=train_epoch_loss,width=3))


PATH = "./Checkpoint/translate_en_de.pth"
torch.save(model.state_dict(), PATH)




