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

def build_model(device, model_params, dec_start_tokenid, pad_token_id, lr, resume_file = None, modelonly_resume = False, freeze_encOnly = False, freeze_decEmbedOnly = False, skipScheduler=False ):
    ### Build Model 
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_params['modelpath_from'], model_params['modelpath_to'])

    model.config.decoder_start_token_id = dec_start_tokenid 
    model.config.pad_token_id = pad_token_id 
    model = model.to(device)

    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad_(False)

    freeze_module(model.encoder)
    if not freeze_encOnly:
        freeze_module(model.decoder.bert.embeddings)
        if not freeze_decEmbedOnly:
            for layer in model.decoder.bert.encoder.layer:
                freeze_module(layer.attention)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = None
    if not skipScheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

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
 

def show_results(model, fromLang_tokenizer, toLang_tokenizer, data_loader, limit = 20):
    model.eval()
    device = model.device
    count = 0
    for i, data in enumerate(itertools.islice(data_loader, limit)):
        out = model.generate(data['lang_from']['input_ids'].to(device))
        predictions = zip(list(map(toLang_tokenizer.decode, out)), 
                          list(map(toLang_tokenizer.decode, data['lang_to']['input_ids'])),
                          list(map(fromLang_tokenizer.decode, data['lang_from']['input_ids'])))
        for p in predictions:
            count += 1
            print(*p, sep='\n')
            print()
        if count>=limit:
            break


def train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, scheduler_freq, epoch, log_writer = None, x0=0, running_loss=0):
    model.train()
    epoch_loss = 0
    
    device = model.device

    num_train_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f'epoch {epoch}', leave=False)
    for i, data in enumerate(pbar):
        step = x0 + epoch * len(train_loader) + i + 1
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

        predictions = out[1]
        # predictions = F.log_softmax(predictions, dim=-1)
        loss = loss_fn(predictions[:, :-1, :].contiguous().view(-1, predictions.shape[-1]), toLang_input_ids[:, 1:].contiguous().view(-1))

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        epoch_loss += loss.item()

        if (step-x0)%scheduler_freq == 0:
            if scheduler:
                scheduler.step(running_loss/scheduler_freq)
            pbar.set_postfix(loss=f'{running_loss/scheduler_freq:.2f}', lr = f"{optimizer.param_groups[0]['lr']:.1e}")
            running_loss = 0

        if log_writer:
            log_writer.add_scalar('training loss', loss.item(), step)

    return epoch_loss / num_train_batches, running_loss


def train_model(model, optimizer, scheduler, scheduler_freq, loss_fn, train_loader, train_params, x0):
    writer = SummaryWriter(train_params['save_prefix'])

    ## Train
    running_loss = 0
    for epoch in range(train_params['num_epochs']):
        train_epoch_loss, running_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, scheduler_freq, epoch, log_writer=writer, x0=x0, running_loss=running_loss)
        print("epoch {epoch:>{width}}: train_loss {train_loss:.3f}".format(epoch=epoch,train_loss=train_epoch_loss,width=3))

        model_savepath = os.path.join(train_params['save_prefix'],train_params['save_format']).format(epoch,'N')
        # save_model(x0 + epoch * len(train_loader), model, optimizer, scheduler, model_savepath)
        old_files = natsorted(glob.glob(os.path.join(train_params['save_prefix'],train_params['save_format'].format('*','*'))))[:-1]
        list(map(os.remove,old_files))

    writer.close()

def main(model_params, data_params, train_params, mode):
    device = torch.device(train_params['device'] if torch.cuda.is_available() else "cpu")

    os.makedirs(train_params['save_prefix'], exist_ok = True)
    torch.save([model_params, data_params, train_params],os.path.join(train_params['save_prefix'],'params.pth'))

    ### Tokenizers
    fromLang_tokenizer = BertTokenizer.from_pretrained(model_params['modelpath_from'])
    toLang_tokenizer = AutoTokenizer.from_pretrained(model_params['modelpath_to'])

    ## Train Data Loader
    train_dataset = TransformerDataset(data_params, 'train', fromLang_tokenizer, toLang_tokenizer, max_len = train_params['maxlen'], subset = train_params['subset'])
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=train_params['batch_size'], 
                                               shuffle=True, 
                                               collate_fn = lambda b:collate_tokens(b, (fromLang_tokenizer, data_params['lang'][0]), (toLang_tokenizer, data_params['lang'][1])))

    loss_fn = nn.CrossEntropyLoss(ignore_index=toLang_tokenizer.pad_token_id)

    if mode == 'train':
        resume_file = natsorted(glob.glob(os.path.join(train_params['resume_dir'],train_params['save_format'].format('*','*'))))[-1] if train_params['resume_dir'] else None
        model, optimizer, scheduler, x0 = build_model(device, model_params, 
                                                      toLang_tokenizer.cls_token_id, toLang_tokenizer.pad_token_id, 
                                                      train_params['lr'], 
                                                      resume_file = resume_file, 
                                                      modelonly_resume = train_params['modelonly_resume'],
                                                      freeze_encOnly = train_params['freeze_encOnly'],
                                                      freeze_decEmbedOnly = train_params['freeze_decEmbedOnly'],
                                                      skipScheduler = train_params['skipScheduler'])
        train_model(model, optimizer, scheduler, train_params['scheduler_freq'], loss_fn, train_loader, train_params, x0)
    else:
        resume_file = natsorted(glob.glob(os.path.join(train_params['save_prefix'],train_params['save_format'].format('*','*'))))[-1] if train_params['save_prefix'] else None
        model, _, _, _ = build_model(device, model_params, 
                                     toLang_tokenizer.cls_token_id, toLang_tokenizer.pad_token_id, 
                                     train_params['lr'], 
                                     resume_file = resume_file,
                                     modelonly_resume = True, 
                                     freeze_encOnly = train_params['freeze_encOnly'],
                                     freeze_decEmbedOnly = train_params['freeze_decEmbedOnly'],
                                     skipScheduler = True)

    test_dataset = TransformerDataset(data_params, 'test', fromLang_tokenizer, toLang_tokenizer, max_len = train_params['maxlen'], subset = train_params['subset_eval'])
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=train_params['batch_size'], 
                                               shuffle=True, 
                                               collate_fn = lambda b:collate_tokens(b, (fromLang_tokenizer, data_params['lang'][0]), (toLang_tokenizer, data_params['lang'][1])))
    show_results(model, fromLang_tokenizer, toLang_tokenizer, test_loader)


if __name__ == '__main__':

    model_params = {
                    'modelpath_from' :  "bert-base-cased", 
                    'modelpath_to' : "dbmdz/bert-base-german-cased"
                 }

    data_params = { 
                'dataset_path' : ('online',('wmt14', 'de-en')),
                'lang' : ('en','de') #(from-language, to-language)
                }

    train_params = {
                    'freeze_encOnly' : False,
                    'freeze_decEmbedOnly' : False,

                    'maxlen' : 76,
                    'subset' : 25000,

                    'batch_size' : 72,
                    'num_epochs' : 100,
                    'subset_eval' : None,
                    'lr' : 2e-4,
                    'scheduler_freq' : 200,
                    'skipScheduler' : False,
                    'device' : 'cuda:0',

                    'save_prefix' : 'runs/en_de/log4',
                    'resume_dir' : 'runs/en_de/log3_maxlen76_subset25k',
                    'modelonly_resume' : False,

                    'save_format' : "translator_epoch_{}_batch_{}.pth"
                    }
    train_params['save_prefix'] += '_encOnly' if train_params['freeze_encOnly'] else ''
    train_params['save_prefix'] += '_maxlen' + str(train_params['maxlen'])
    train_params['save_prefix'] += '_subset' + str(int(train_params['subset']/1000)) + 'k'
    
    parser = argparse.ArgumentParser(description='translator_{}_{}'.format(*data_params['lang']))

    parser.add_argument('mode', choices = ['train','eval'])
    for k in train_params.keys():
        if isinstance(train_params[k], bool):
            parser.add_argument('--'+k, default=train_params[k], action='store_true')
        else:
            parser.add_argument('--'+k, default=train_params[k], type=type(train_params[k]))

    args = parser.parse_args()
    for k in train_params.keys():
        train_params[k] = getattr(args, k)

    main(model_params, data_params, train_params, args.mode)