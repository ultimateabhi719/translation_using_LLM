#!/usr/bin/env fish


# smallTrain
python src/translator.py  train --device cuda:0 --maxlen 100 \
                                                --subset 25000 \
                                                --batch_size 72 \
                                                --num_epochs 200 \
                                                --save_prefix runs/en_de/smallTrain \
                                                --resume_dir None \
                                                --scheduler_freq 200

# FullTrain
python src/translator.py  train --device cuda:0 --maxlen 50 \
                                                --subset 0 \
                                                --batch_size 72 \
                                                --num_epochs 20 \
                                                --save_prefix runs/en_de/log1 \
                                                --resume_dir runs/en_de/log0_maxlen50_subset0/ \
                                                --modelOnly_resume \
                                                --scheduler_freq 200 \
                                                --min_lr 2e-6

## NER
# overfit
python src/translator_v1.py  train --device cuda:1 --maxlen 50 \
                                               --subset 240 \
                                               --batch_size 72 \
                                               --num_epochs 50 \
                                               --save_prefix None \
                                               --resume_dir None \
                                               --init_lr 2e-4

# fit small train
python src/translator_v1.py  train --device cuda:1 --maxlen 50 \
                                               --subset 25000 \
                                               --batch_size 72 \
                                               --num_epochs 50 \
                                               --save_prefix runs/en_de/nerSmallTrain \
                                               --resume_dir None \
                                               --init_lr 2e-4

# FullTrain
python src/translator_v1.py  train --device cuda:1 --maxlen 50 \
                                               --subset 0 \
                                               --batch_size 200 \
                                               --num_epochs 50 \
                                               --save_prefix runs/en_de/ner_log0 \
                                               --resume_dir runs/en_de/nerSmallTrain_maxlen50_subset25k/ \
                                               --init_lr 2e-4

python src/translator_v1.py  eval --device cuda:0 --maxlen 50 \
                                               --subset 0 \
                                               --batch_size 72 \
                                               --num_epochs 50 \
                                               --save_prefix None \
                                               --resume_dir runs/en_de/ner_log0_maxlen50_subset0/ \
                                               --init_lr 2e-4

