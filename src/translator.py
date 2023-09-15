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
from torchmetrics.text import BLEUScore

from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, BertTokenizer, EncoderDecoderModel
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import collate_tokens, save_model
from dataset import TransformerDataset


def build_model(model_params, device, resume_file = None):
    ## Tokenizers
    fromLang_tokenizer = BertTokenizer.from_pretrained(model_params['modelpath_from'])
    toLang_tokenizer = AutoTokenizer.from_pretrained(model_params['modelpath_to'])

    ## Build Model 
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_params['modelpath_from'], model_params['modelpath_to'])

    model.config.decoder_start_token_id = toLang_tokenizer.cls_token_id 
    model.config.pad_token_id = toLang_tokenizer.pad_token_id 
    model = model.to(device)

    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad_(False)

    freeze_module(model.encoder)
    if not model_params['freeze_encOnly']:
        freeze_module(model.decoder.bert.embeddings)
        if not model_params['freeze_decEmbedOnly']:
            for layer in model.decoder.bert.encoder.layer:
                freeze_module(layer.attention)
    else:
        assert not model_params['freeze_decEmbedOnly'], "specify only one of `freeze_decEmbedOnly` or `freeze_encOnly`"

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=model_params['init_lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=model_params['gamma'], patience=model_params['patience'], min_lr= model_params['min_lr'],verbose=True)

    ## Load model and optimizer from resume_dir
    x0 = 0
    if resume_file:
        print(f"LOADING init model from {resume_file}..")
        assert os.path.exists(resume_file)
        resume_dict = torch.load(resume_file, map_location=device) 

        model.load_state_dict(resume_dict['model_state_dict'])
        if not model_params['modelOnly_resume']:
            optimizer.load_state_dict(resume_dict['optimizer_state_dict'])
            scheduler.load_state_dict(resume_dict['scheduler_state_dict'])
        x0 = resume_dict['x0']
    print(f"init LEARNING RATE: {optimizer.param_groups[0]['lr']}")


    return fromLang_tokenizer, toLang_tokenizer, model, optimizer, scheduler, x0


@torch.no_grad()
def eval(model, test_loader, maxlen, toLang_tokenizer, nbeams=20, num_returns=3, limit=20):
    bleu = BLEUScore(n_gram=3)

    count = 0
    bleu_score = 0
    for i, data in enumerate(tqdm(test_loader, desc=f'eval', leave=False)):
        out = model.generate(data['lang_from']['input_ids'].to(model.device), max_length=maxlen, num_beams=nbeams, num_return_sequences=num_returns, do_sample=False)

        sep_idx = torch.argmax((out == toLang_tokenizer.sep_token_id).to(dtype=torch.int), dim=-1).tolist()
        out_text = []
        for i, idx in enumerate(sep_idx):
            out_text.append(toLang_tokenizer.decode(out[i,1:idx]))

        target = data['lang_to']['input_ids']
        sep_idx = torch.argmax((target == toLang_tokenizer.sep_token_id).to(dtype=torch.int), dim=-1).tolist()

        for idx in range(target.shape[0]):
            target_idx = toLang_tokenizer.decode(target[idx,1:sep_idx[idx]])
            out_idx = out_text[idx*num_returns:(idx+1)*num_returns]
            bleu_score += bleu(out_idx,[[target_idx]])
            if count<=limit:
                print(target_idx)
                print(out_idx)
                print()
            count += 1
    return bleu_score/len(test_loader)


def train_one_epoch(model, optimizer, scheduler, scheduler_freq, loss_fn, train_loader, epoch, save_freq, log_writer = None, x0=0, running_loss=0, model_savefmt=None):
    model.train()
    epoch_loss = 0
    device = model.device

    num_train_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f'epoch {epoch}', leave=False)
    pbar.set_postfix(loss=f'{running_loss/scheduler_freq:.2f}', lr = f"{optimizer.param_groups[0]['lr']:.1e}")
    for i, data in enumerate(pbar):
        step = x0 + epoch * len(train_loader) + i + 1
        
        in_tokens = data['lang_from']['input_ids'].to(device)
        in_masks = data['lang_from']['attention_mask'].to(device)
        
        out_tokens = data['lang_to']['input_ids'].to(device)
        out_masks = data['lang_to']['attention_mask'].to(device)

        optimizer.zero_grad()
        # decoder_input_ids = shift_tokens_right(out_tokens, model.config.pad_token_id, model.config.decoder_start_token_id).to(device)
        out = model(input_ids=in_tokens, attention_mask = in_masks, decoder_input_ids=out_tokens,labels=out_tokens,decoder_attention_mask=out_masks)
        predictions = out[1]
        # predictions = F.log_softmax(predictions, dim=-1)
        loss = loss_fn(predictions[:, :-1, :].contiguous().view(-1, predictions.shape[-1]), out_tokens[:, 1:].contiguous().view(-1))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()

        if (step-x0)%scheduler_freq == 0:
            scheduler.step(running_loss/scheduler_freq)
            pbar.set_postfix(loss=f'{running_loss/scheduler_freq:.2f}', lr = f"{optimizer.param_groups[0]['lr']:.1e}")
            running_loss = 0

        if model_savefmt and (step-x0)%save_freq == 0:
            save_model(step, model, optimizer, scheduler, model_savefmt.format(epoch,i+1))
            old_files = natsorted(glob.glob(model_savefmt.format('*','*')))[:-1]
            list(map(os.remove,old_files))

        if log_writer:
            log_writer.add_scalar('training loss', loss.item(), step)

    return epoch_loss / num_train_batches, running_loss


def main(mode, model_params, data_params, train_params):
    model_savefmt = os.path.join(train_params['save_prefix'],train_params['save_format']) if train_params['save_prefix'] else None
    if train_params['save_prefix']:
        os.makedirs(train_params['save_prefix'], exist_ok = True)
        torch.save([model_params, data_params, train_params],os.path.join(train_params['save_prefix'],'params.pth'))

    ## Build Model
    device = torch.device(train_params['device'] if torch.cuda.is_available() else "cpu")
    resume_file = natsorted(glob.glob(os.path.join(train_params['resume_dir'],train_params['save_format'].format('*','*'))))[-1] if train_params['resume_dir'] else None
    fromLang_tokenizer, toLang_tokenizer, model, optimizer, scheduler, x0 = build_model(model_params, device, resume_file=resume_file)
    x0 = x0 if train_params['x0']<0 else train_params['x0']

    ## Data
    train_dataset = TransformerDataset(data_params, 'train', fromLang_tokenizer, toLang_tokenizer, max_len = train_params['maxlen'], subset = train_params['subset'])
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=train_params['batch_size'], 
                                               shuffle=True, 
                                               collate_fn = lambda b:collate_tokens(b, (fromLang_tokenizer, data_params['lang'][0]), (toLang_tokenizer, data_params['lang'][1])))
    test_dataset = TransformerDataset(data_params, 'test', fromLang_tokenizer, toLang_tokenizer, max_len = train_params['maxlen'], subset = train_params['subset_eval'])
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=train_params['batch_size']//8, 
                                               shuffle=True, 
                                               collate_fn = lambda b:collate_tokens(b, (fromLang_tokenizer, data_params['lang'][0]), (toLang_tokenizer, data_params['lang'][1])))

    ## Train
    if mode == 'train':
        writer = SummaryWriter(train_params['save_prefix']) if train_params['save_prefix'] else None

        loss_fn = nn.CrossEntropyLoss(ignore_index=toLang_tokenizer.pad_token_id)
        running_loss = 0
        for epoch in range(train_params['num_epochs']):
            train_epoch_loss, running_loss = train_one_epoch(model, optimizer, scheduler, train_params['scheduler_freq'], loss_fn, train_loader, epoch, train_params['save_freq'],
                                                             log_writer=writer, x0=x0, running_loss=running_loss, model_savefmt=model_savefmt)
            print("epoch {epoch:>{width}}: train_loss {train_loss:.3f}".format(epoch=epoch,train_loss=train_epoch_loss,width=3))

            if train_params['save_prefix']:
                save_model(x0 + epoch * len(train_loader), model, optimizer, scheduler, model_savefmt.format(epoch,'N'))
                old_files = natsorted(glob.glob(os.path.join(train_params['save_prefix'],train_params['save_format'].format('*','*'))))[:-1]
                list(map(os.remove,old_files))

        if writer:
            writer.close()
            
    ## Eval
    print("bleu score", eval(model, test_loader, train_params['maxlen'], toLang_tokenizer, nbeams=20, num_returns=3))

if __name__ == "__main__":
    model_params = {
                    # model options
                    'modelpath_from' :  "bert-base-cased", 
                    'modelpath_to' : "dbmdz/bert-base-german-cased",

                    'freeze_encOnly' : False,
                    'freeze_decEmbedOnly' : False,
                    'skipScheduler' : False,
                    'modelOnly_resume' : False,

                    # optimizer & schduler options
                    'init_lr' : 2e-4,
                    'min_lr' : 1e-6,
                    'gamma' : 0.5,
                    'patience' : 2
                 }

    data_params = { 
                'dataset_path' : ('online',('wmt14', 'de-en')),
                'lang' : ('en','de') #(from-language, to-language)
                }

    train_params = {
                    # data subsetting options
                    'maxlen' : 76,
                    'subset' : 288,
                    'subset_eval' : 0,

                    # training options
                    'batch_size' : 24,
                    'num_epochs' : 30,
                    'scheduler_freq' : 12,
                    'device' : 'cuda:0',

                    # checkpoint options
                    'save_prefix' : None,#'runs/en_de/overfit',
                    'resume_dir' : None,#'runs/en_de/log3_maxlen76_subset25k',
                    'save_freq' : 1000,
                    'x0' : -1,
                    'save_format' : "translator_epoch_{}_batch_{}.pth",
                    }

    parser = argparse.ArgumentParser(description='translator_{}_{}'.format(*data_params['lang']))
    parser.add_argument('mode', choices = ['train','eval'])
    for k in train_params.keys():
        if isinstance(train_params[k], bool):
            parser.add_argument('--'+k, default=train_params[k], action='store_true')
        elif isinstance(train_params[k], type(None)) or isinstance(train_params[k], str):
            parser.add_argument('--'+k, default=train_params[k], type=lambda x : None if x == 'None' else str(x))
        elif k=='subset':
            parser.add_argument('--'+k, default=train_params[k], type=lambda x : None if x == 'None' else int(x))
        else:
            parser.add_argument('--'+k, default=train_params[k], type=type(train_params[k]))
    for k in model_params.keys():
        if isinstance(model_params[k], bool):
            parser.add_argument('--'+k, default=model_params[k], action='store_true')
        elif isinstance(model_params[k], type(None)) or isinstance(model_params[k], str):
            parser.add_argument('--'+k, default=model_params[k], type=lambda x : None if x == 'None' else str(x))
        else:
            parser.add_argument('--'+k, default=model_params[k], type=type(model_params[k]))


    args = parser.parse_args()
    model_params['modelOnly_resume'] = (args.mode == 'eval')
    for k in train_params.keys():
        train_params[k] = getattr(args, k)
    for k in model_params.keys():
        model_params[k] = getattr(args, k)

    # add stuff to save_prefix
    if train_params['save_prefix']:
        train_params['save_prefix'] += '_encOnly' if model_params['freeze_encOnly'] else '_decEmbedOnly' if model_params['freeze_decEmbedOnly'] else ''
        train_params['save_prefix'] += '_maxlen' + str(train_params['maxlen'])
        train_params['save_prefix'] += '_subset' + ((str(int(train_params['subset']/1000)) + 'k') if train_params['subset']>999 else str(train_params['subset']))

    print("model RUNDIR:",train_params['save_prefix'])
    
    main(args.mode, model_params, data_params, train_params)
