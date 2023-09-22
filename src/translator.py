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
from transformers import AutoTokenizer, BertTokenizer, EncoderDecoderModel, AutoModelForTokenClassification

from utils import collate_tokens, save_model
from dataset import TransformerDataset

## Transformer
class Translator(nn.Module):
    def __init__(self, model_params, toLang_tokenizer):
        super(Translator, self).__init__()
        ## Build Enc-Dec Model 
        self.enc_dec = EncoderDecoderModel.from_encoder_decoder_pretrained(model_params['modelpath_from'], model_params['modelpath_to'])

        self.enc_dec.config.decoder_start_token_id = toLang_tokenizer.cls_token_id 
        self.enc_dec.config.pad_token_id = toLang_tokenizer.pad_token_id 

        # Build NER model
        self.ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        def freeze_module(module):
            for param in module.parameters():
                param.requires_grad_(False)

        # freeze layers of Encoder Decoder Model
        freeze_module(self.enc_dec.encoder)
        if not model_params['freeze_encOnly']:
            freeze_module(self.enc_dec.decoder.bert.embeddings)
            if not model_params['freeze_decEmbedOnly']:
                for layer in self.enc_dec.decoder.bert.encoder.layer:
                    freeze_module(layer.attention)
        else:
            assert not model_params['freeze_decEmbedOnly'], "specify only one of `freeze_decEmbedOnly` or `freeze_encOnly`"

        # freeze NER model
        freeze_module(self.ner)

        # Linear Layer to shrink Encoder LLM output
        self.shrink_enc_out = nn.Linear(self.enc_dec.encoder.config.hidden_size, 
                                        self.enc_dec.decoder.config.hidden_size-len(self.ner.config.label2id))

    @torch.no_grad()
    def decode(self, input_ids, attention_mask, max_length=25):
        """
        for inference
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out_labels : returns final prediction of sequence
        """
        
        input_device = input_ids.device

        ner_classes = F.softmax(self.ner(input_ids).logits, dim=-1)

        enc_output = self.enc_dec.encoder(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          return_dict=True)[0]
        shrunk_enc_out = self.shrink_enc_out(enc_output)

        encoder_hidden_states_extended = torch.cat((ner_classes, shrunk_enc_out), dim=-1)

        out_labels = torch.tensor([[self.enc_dec.config.decoder_start_token_id]]*input_ids.shape[0]).to(input_device)
        for _ in range(max_length):
            dec_output = self.enc_dec.decoder(input_ids = out_labels, 
                                              encoder_hidden_states=encoder_hidden_states_extended, 
                                              encoder_attention_mask=attention_mask, 
                                              return_dict=True).logits

            out_labels = torch.cat((out_labels, dec_output[:,-1:,:].argmax(-1)),dim=1)

        return out_labels

    def forward(self, input_ids, attention_mask, decoder_input_ids,decoder_attention_mask):

        ner_classes = F.softmax(self.ner(input_ids).logits, dim=-1)

        encoder_hidden_states = self.enc_dec.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True
        )[0]

        # shrunk enc output and add one hot encoding of NER classes
        shrunk_enc_out = self.shrink_enc_out(encoder_hidden_states)
        encoder_hidden_states_extended = torch.cat((ner_classes, shrunk_enc_out), dim=-1)

        decoder_outputs = self.enc_dec.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states_extended,
            encoder_attention_mask=attention_mask,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            use_cache=None,
            past_key_values=None,
            return_dict=True
        )

        return decoder_outputs.logits


@torch.no_grad()
def eval(model, test_loader, maxlen, toLang_tokenizer, limit=20):
    model.eval()
    f = lambda color: lambda string, **kwargs: print(color + string + "\33[0m", **kwargs)
    p_red = f("\33[31m")

    device_model = next(model.parameters()).device
    count = 0
    bleu_score = 0
    max_ngrams = 3
    bleu = BLEUScore(n_gram=max_ngrams)
    for i, data in enumerate(tqdm(test_loader, desc=f'eval', leave=False)):
        out = model.decode(input_ids=data['lang_from']['input_ids'].to(device_model), attention_mask=data['lang_from']['attention_mask'].to(device_model), max_length=maxlen)

        sep_idx = torch.argmax((out == toLang_tokenizer.sep_token_id).to(dtype=torch.int), dim=-1).tolist()
        out_text = []
        for i, idx in enumerate(sep_idx):
            out_text.append(toLang_tokenizer.decode(out[i,1:idx]))

        target = data['lang_to']['input_ids']
        sep_idx = torch.argmax((target == toLang_tokenizer.sep_token_id).to(dtype=torch.int), dim=-1).tolist()

        for idx in range(target.shape[0]):
            target_idx = toLang_tokenizer.decode(target[idx,1:sep_idx[idx]])
            if len(target_idx.split())>=max_ngrams:
                out_idx = out_text[idx]
                bleu_idx = bleu([out_idx],[[target_idx]])
                bleu_score += bleu_idx
                if count<=limit:
                    print(target_idx)
                    print(out_idx)
                    print()
                count += 1
    return bleu_score/count


def build_model(model_params, device, resume_file = None):
    ## Tokenizers
    fromLang_tokenizer = BertTokenizer.from_pretrained(model_params['modelpath_from'])
    toLang_tokenizer = AutoTokenizer.from_pretrained(model_params['modelpath_to'])

    model = Translator(model_params, toLang_tokenizer).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=model_params['init_lr'])

    ## Load model and optimizer from resume_dir
    x0 = 0
    if resume_file:
        print(f"LOADING init model from {resume_file}..")
        assert os.path.exists(resume_file)
        resume_dict = torch.load(resume_file, map_location=device) 

        model.load_state_dict(resume_dict['model_state_dict'])
        if not model_params['modelOnly_resume']:
            optimizer.load_state_dict(resume_dict['optimizer_state_dict'])
        x0 = resume_dict['x0']
    print(f"init LEARNING RATE: {optimizer.param_groups[0]['lr']}")

    return fromLang_tokenizer, toLang_tokenizer, model, optimizer, x0



def train_one_epoch(model, optimizer, loss_fn, train_loader, epoch, save_freq, log_writer = None, x0=0, model_savefmt=None):
    model.train()
    epoch_loss = 0
    device = next(model.parameters()).device

    num_train_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f'epoch {epoch}', leave=False)
    for i, data in enumerate(pbar):
        step = x0 + epoch * len(train_loader) + i + 1
        
        in_tokens = data['lang_from']['input_ids'].to(device)
        in_masks = data['lang_from']['attention_mask'].to(device)
        
        out_tokens = data['lang_to']['input_ids'].to(device)
        out_masks = data['lang_to']['attention_mask'].to(device)

        optimizer.zero_grad()
        predictions = model(input_ids=in_tokens, attention_mask = in_masks, decoder_input_ids=out_tokens,decoder_attention_mask=out_masks)
        loss = loss_fn(predictions[:, :-1, :].contiguous().view(-1, predictions.shape[-1]), out_tokens[:, 1:].contiguous().view(-1))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.2f}')

        if model_savefmt and (step-x0)%save_freq == 0:
            save_model(step, model, optimizer, model_savefmt.format(epoch,i+1))
            old_files = natsorted(glob.glob(model_savefmt.format('*','*')))[:-1]
            list(map(os.remove,old_files))

        if log_writer:
            log_writer.add_scalar('training loss', loss.item(), step)

    return epoch_loss / num_train_batches


def main(mode, model_params, data_params, train_params):
    model_savefmt = os.path.join(train_params['save_prefix'],train_params['save_format']) if train_params['save_prefix'] else None
    if train_params['save_prefix']:
        os.makedirs(train_params['save_prefix'], exist_ok = True)
        torch.save([model_params, data_params, train_params],os.path.join(train_params['save_prefix'],'params.pth'))

    ## Build Model
    device = torch.device(train_params['device'] if torch.cuda.is_available() else "cpu")
    resume_file = natsorted(glob.glob(os.path.join(train_params['resume_dir'],train_params['save_format'].format('*','*'))))[-1] if train_params['resume_dir'] else None
    fromLang_tokenizer, toLang_tokenizer, model, optimizer, x0 = build_model(model_params, device, resume_file=resume_file)
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
        for epoch in range(train_params['num_epochs']):
            train_epoch_loss = train_one_epoch(model, optimizer, loss_fn, train_loader, epoch, train_params['save_freq'],
                                                             log_writer=writer, x0=x0, model_savefmt=model_savefmt)
            print("epoch {epoch:>{width}}: train_loss {train_loss:.3f}".format(epoch=epoch,train_loss=train_epoch_loss,width=3))

            if train_params['save_prefix']:
                save_model(x0 + epoch * len(train_loader), model, optimizer, model_savefmt.format(epoch,'N'))
                old_files = natsorted(glob.glob(os.path.join(train_params['save_prefix'],train_params['save_format'].format('*','*'))))[:-1]
                list(map(os.remove,old_files))

        if writer:
            writer.close()
    
    ## Eval
    print("bleu score:", eval(model, test_loader, train_params['maxlen'], toLang_tokenizer))

if __name__ == "__main__":
    model_params = {
                    # model options
                    'modelpath_from' :  "bert-base-cased", 
                    'modelpath_to' : "dbmdz/bert-base-german-cased",

                    'freeze_encOnly' : False,
                    'freeze_decEmbedOnly' : False,
                    'modelOnly_resume' : False,

                    # optimizer options
                    'init_lr' : 2e-4,
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
