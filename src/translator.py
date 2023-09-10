#!/usr/bin/env python
# coding: utf-8

import os
import glob
import argparse
import itertools
from natsort import natsorted
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, BertTokenizer, EncoderDecoderModel
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import collate_tokens, save_model
from dataset import TransformerDataset

def train_one_epoch(model, device, train_loader, loss_fn, optimizer, epoch, log_writer = None, x0 = 0):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        if log_writer:
            step = x0 + epoch * len(train_loader) + i + 1
            log_writer.add_scalar('training loss', loss.item(), step)

    return epoch_loss / num_train_batches


def build_model(device, modelpath_from, modelpath_to, dec_start_tokenid, pad_token_id, lr, resume_file = None, modelonly_resume = False):
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0, verbose=True)

    ## Load model and optimizer from resume_dir
    x0 = 0
    if resume_file:
        print(f"LOADING init model from {resume_file}..")
        assert os.path.exists(resume_file)
        resume_dict = torch.load(resume_file, map_location=device) 

        model.load_state_dict(resume_dict['model_state_dict'])
        if not modelonly_resume:
            optimizer.load_state_dict(resume_dict['optimizer_state_dict'])
            if 'scheduler_state_dict' in resume_dict.keys():
                scheduler.load_state_dict(resume_dict['scheduler_state_dict'])
        x0 = resume_dict['x0']
    print(f"init LEARNING RATE: {optimizer.param_groups[0]['lr']}")

    return model, optimizer, scheduler, x0
 

def show_results(model, device, fromLang_tokenizer, toLang_tokenizer, data_loader, limit = 10):
    model.eval()
    for i, data in enumerate(itertools.islice(data_loader, limit)):
        out = model.generate(data['lang_from']['input_ids'].to(device))
        predictions = zip(list(map(toLang_tokenizer.decode, out)), 
                          list(map(toLang_tokenizer.decode, data['lang_to']['input_ids'])),
                          list(map(fromLang_tokenizer.decode, data['lang_from']['input_ids'])))
        for p in predictions:
            print(*p, sep='\n')
            print()


def train_model(device, train_params, train_loader, loss_fn, dec_start_tokenid, pad_token_id):

    resume_file = natsorted(glob.glob(os.path.join(train_params['resume_dir'],train_params['save_format'].format('*','*'))))[-1] if train_params['resume_dir'] else None
    model, optimizer, scheduler, x0 = build_model(device, train_params['modelpath_from'], train_params['modelpath_to'], dec_start_tokenid, pad_token_id, train_params['lr'], 
                                                  resume_file = resume_file, modelonly_resume = train_params['modelonly_resume'])

    writer = SummaryWriter(train_params['save_prefix'])

    ## Train
    for epoch in range(train_params['num_epochs']):
        train_epoch_loss = train_one_epoch(model, device, train_loader, loss_fn, optimizer, epoch, log_writer=writer, x0=x0)

        model_savepath = os.path.join(train_params['save_prefix'],train_params['save_format']).format(epoch,'N')
        save_model(epoch * len(train_loader), model, optimizer, scheduler, model_savepath)
 
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(train_epoch_loss)
        after_lr = optimizer.param_groups[0]["lr"]
        print("epoch {epoch:>{width}}: train_loss {train_loss:.3f}".format(epoch=epoch,train_loss=train_epoch_loss,width=3))
     
    writer.close()

    return model

def main(data_params, train_params, mode):
    device = torch.device(train_params['device'] if torch.cuda.is_available() else "cpu")

    os.makedirs(train_params['save_prefix'], exist_ok = True)

    ### Tokenizers
    fromLang_tokenizer = BertTokenizer.from_pretrained(train_params['modelpath_from'])
    toLang_tokenizer = AutoTokenizer.from_pretrained(train_params['modelpath_to'])

    ## Train Data Loader
    train_dataset = TransformerDataset(data_params, 'train', fromLang_tokenizer, toLang_tokenizer, max_len = 100, subset = train_params['subset'])
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=train_params['batch_size'], 
                                               shuffle=True, 
                                               collate_fn = lambda b:collate_tokens(b, (fromLang_tokenizer, data_params['lang'][0]), (toLang_tokenizer, data_params['lang'][1])))

    loss_fn = nn.CrossEntropyLoss(ignore_index=toLang_tokenizer.pad_token_id)

    if mode == 'train':
        model = train_model(device, train_params, train_loader, loss_fn, toLang_tokenizer.cls_token_id,  toLang_tokenizer.pad_token_id)
    else:
        resume_file = natsorted(glob.glob(os.path.join(train_params['save_prefix'],train_params['save_format'].format('*','*'))))[-1] if train_params['save_prefix'] else None
        model, _, _, _ = build_model(device, train_params['modelpath_from'], train_params['modelpath_to'], 
                                       toLang_tokenizer.cls_token_id,  toLang_tokenizer.pad_token_id, 
                                       train_params['lr'], 
                                       resume_file = resume_file)

    test_dataset = TransformerDataset(data_params, 'test', fromLang_tokenizer, toLang_tokenizer, max_len = 100, subset = train_params['subset_eval'])
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=train_params['batch_size'], 
                                               shuffle=True, 
                                               collate_fn = lambda b:collate_tokens(b, (fromLang_tokenizer, data_params['lang'][0]), (toLang_tokenizer, data_params['lang'][1])))
    show_results(model, device, fromLang_tokenizer, toLang_tokenizer, test_loader)


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
                    'lr' : 1e-3,
                    'device' : 'cuda:0',

                    'modelpath_from' :  "bert-base-cased", 
                    'modelpath_to' : "dbmdz/bert-base-german-cased",

                    'save_prefix' : 'runs/freeze_enc_part_dec/en_de_maxlen100_subset100k_log2',
                    'resume_dir' : 'runs/freeze_enc_part_dec/en_de_maxlen100_subset100k_lr3e-4_log1',
                    'save_format' : "translator_epoch_{}_batch_{}.pth",
                    'modelonly_resume' : False
                    }
    
    parser = argparse.ArgumentParser(description='translator_{}_{}'.format(*data_params['lang']))

    parser.add_argument('mode', choices = ['train','eval'])
    for k in train_params.keys():
        parser.add_argument('--'+k, default=train_params[k], type=type(train_params[k]))

    args = parser.parse_args()
    for k in train_params.keys():
        train_params[k] = getattr(args, k)

    main(data_params, train_params, args.mode)