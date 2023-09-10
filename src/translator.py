#!/usr/bin/env python
# coding: utf-8

import os
import glob
import argparse
from natsort import natsorted
import itertools

import pandas as pd

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from tqdm.auto import tqdm

from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, BertTokenizer, EncoderDecoderModel

from utils import collate_tokens, save_model
from dataset import TransformerDataset

def train_one_epoch(model, device, train_loader, loss_fn, optimizer, epoch):
    model.train()
    epoch_loss = 0

    num_train_batches = len(train_loader)
    for i, data in enumerate(tqdm(train_loader, desc=f'epoch {epoch}', leave=False)):

        optimizer.zero_grad()

        fromLang_input_ids = data['lang_from']['input_ids'].to(device)
        fromLang_masks = data['lang_from']['attention_mask'].to(device)
        
        toLang_input_ids = data['lang_to']['input_ids'].to(device)
        toLang_masks = data['lang_to']['attention_mask'].to(device)

        out = model(input_ids=fromLang_input_ids, 
                    attention_mask = fromLang_masks, 
                    decoder_input_ids=toLang_input_ids,
                    labels=toLang_input_ids,
                    decoder_attention_mask=toLang_masks)

        prediction_scores = out[1]
        predictions = F.log_softmax(prediction_scores, dim=-1)
        loss = loss_fn(predictions[:, :-1, :].contiguous().view(-1, predictions.shape[-1]), toLang_input_ids[:, 1:].contiguous().view(-1))

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / num_train_batches


def build_model(device, modelpath_from, modelpath_to, dec_start_tokenid, pad_token_id, lr, resume_file = None):
    ### Build Model 
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(modelpath_from, modelpath_to)

    model.config.decoder_start_token_id = dec_start_tokenid 
    model.config.pad_token_id = pad_token_id 
    model = model.to(device)

    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad_(False)

    freeze_module(model.encoder)
    freeze_module(model.decoder.bert.embeddings)
    for layer in model.decoder.bert.encoder.layer:
        freeze_module(layer.attention)

    optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    ## Load model and optimizer from resume_dir
    if resume_file:
        print(f"loading init model from {resume_file}..")
        assert os.path.exists(resume_file)
        resume_dict = torch.load(resume_file, map_location=device) 

        model.load_state_dict(resume_dict['model_state_dict'])
        optimizer.load_state_dict(resume_dict['optimizer_state_dict'])

    return model, optimizer
 
def train_model(device, train_params, train_loader, loss_fn, dec_start_tokenid, pad_token_id):

    resume_file = natsorted(glob.glob(os.path.join(train_params['resume_dir'],train_params['save_format'].format('*','*'))))[-1] if train_params['resume_dir'] else None
    model, optimizer = build_model(device, train_params['modelpath_from'], train_params['modelpath_to'], dec_start_tokenid, pad_token_id, train_params['lr'], resume_file = resume_file)

    ## Train
    for epoch in range(train_params['num_epochs']):
        train_epoch_loss = train_one_epoch(model, device, train_loader, loss_fn, optimizer, epoch)
        print("epoch {epoch:>{width}}: train_loss {train_loss:.3f}".format(epoch=epoch,train_loss=train_epoch_loss,width=3))

        model_savepath = os.path.join(train_params['save_prefix'],train_params['save_format']).format(epoch,'N')
        save_model(epoch * len(train_loader), model, optimizer, model_savepath)

    # save_model(epoch * len(train_loader), model, optimizer, model_savepath)


    return model



def show_results(model, fromLang_tokenizer, toLang_tokenizer, data_loader, limit = 10):
    model.eval()
    for i, data in enumerate(itertools.islice(data_loader, limit)):
        out = model.generate(data['lang_from']['input_ids'].to(device))
        predictions = zip(list(map(toLang_tokenizer.decode, out)), 
                          list(map(toLang_tokenizer.decode, data['lang_to']['input_ids'])),
                          list(map(fromLang_tokenizer.decode, data['lang_from']['input_ids'])))
        for p in predictions:
            print(*p, sep='\n')
            print()



if __name__ == '__main__':

    data_params = { 
                'dataset_path' : ('online',('wmt14', 'de-en')),
                'tokenizers':{'from':"bert-base-cased", 'to':"dbmdz/bert-base-german-cased"},
                'lang' : ('en','de') #(from-language, to-language)
                }

    train_params = {
                    'batch_size' : 72,
                    'num_epochs' : 20,
                    'subset' : 100000,
                    'subset_eval' : None,
                    'lr' : 1e-4,
                    'device' : 'cuda:0',

                    'modelpath_from' :  "bert-base-cased", 
                    'modelpath_to' : "dbmdz/bert-base-german-cased",

                    'save_prefix' : 'runs/freeze_encoder_only/en_de_maxlen100_subset100k_lr1e-4_log1',
                    'resume_dir' : 'runs/freeze_encoder_only/en_de_maxlen100_subset100k_lr1e-4',
                    'save_format' : "translator_epoch_{}_batch_{}.pth"
                    }
    
    parser = argparse.ArgumentParser(description='translator_{}_{}'.format(*data_params['lang']))

    parser.add_argument('mode', choices = ['train','eval'])
    for k in train_params.keys():
        parser.add_argument('--'+k, default=train_params[k], type=type(train_params[k]))

    args = parser.parse_args()
    for k in train_params.keys():
        train_params[k] = getattr(args, k)

    device = torch.device(train_params['device'] if torch.cuda.is_available() else "cpu")

    os.makedirs(train_params['save_prefix'], exist_ok = True)

    ### Tokenizers
    fromLang_tokenizer = BertTokenizer.from_pretrained(train_params['modelpath_from'])
    toLang_tokenizer = AutoTokenizer.from_pretrained(train_params['modelpath_to'])

    ## Data Loaders
    train_dataset = TransformerDataset(data_params, 'train', fromLang_tokenizer, toLang_tokenizer, max_len = 100, subset = train_params['subset'])
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=train_params['batch_size'], 
                                               shuffle=True, 
                                               collate_fn = lambda b:collate_tokens(b, (fromLang_tokenizer, data_params['lang'][0]), (toLang_tokenizer, data_params['lang'][1])))

    loss_fn = nn.CrossEntropyLoss(ignore_index=toLang_tokenizer.pad_token_id)

    if args.mode == 'train':
        model = train_model(device, train_params, train_loader, loss_fn, toLang_tokenizer.cls_token_id,  toLang_tokenizer.pad_token_id)
    else:
        resume_file = natsorted(glob.glob(os.path.join(train_params['save_prefix'],train_params['save_format'].format('*','*'))))[-1] if train_params['save_prefix'] else None
        model, _ = build_model(device, train_params['modelpath_from'], train_params['modelpath_to'], 
                                       toLang_tokenizer.cls_token_id,  toLang_tokenizer.pad_token_id, 
                                       train_params['lr'], 
                                       resume_file = resume_file)

    test_dataset = TransformerDataset(data_params, 'test', fromLang_tokenizer, toLang_tokenizer, max_len = 100, subset = train_params['subset_eval'])
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=train_params['batch_size'], 
                                               shuffle=True, 
                                               collate_fn = lambda b:collate_tokens(b, (fromLang_tokenizer, data_params['lang'][0]), (toLang_tokenizer, data_params['lang'][1])))
    show_results(model, fromLang_tokenizer, toLang_tokenizer, test_loader)
